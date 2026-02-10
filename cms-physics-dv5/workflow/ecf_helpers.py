"""
Helper functions for ECF calculation workflow.
Contains preprocessing, file handling, and analysis utilities.
"""

import json
import shutil
import os
import dask
import dask_awkward as dak
import awkward as ak
import numpy as np
from coffea import dataset_tools
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
import fastjet
import scipy


def batch_copy_files(file_list, dst_folder):
    """Copy a list of files to destination folder."""
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    for file_name in file_list:
        if os.path.isfile(file_name):
            shutil.copy(file_name, dst_folder)
            print(f"{file_name} copied to {dst_folder}")
        else:
            print(f"file {file_name} does not exist.")


def preprocess_data(samples_path, step_size=50_000, manager=None):
    """
    Preprocess data files by scanning directory structure and creating metadata.
    
    Parameters:
    -----------
    samples_path : str
        Path to directory containing sample ROOT files
    step_size : int
        Number of events per chunk (default: 50,000)
    manager : DaskVine manager
        TaskVine manager for distributed processing
        
    Returns:
    --------
    dict : Dictionary of preprocessed samples ready for analysis
    """
    print(f"====== Preprocessing data from {samples_path}")
    
    filelist = {}
    categories = os.listdir(samples_path)
    print(f"categories = {categories}")

    # Build file list from directory structure
    for i in categories:
        if '.root' in os.listdir(f'{samples_path}/{i}')[0]:
            files = os.listdir(f'{samples_path}/{i}')
            filelist[i] = [f'{samples_path}/{i}/{file}' for file in files]
        else:
            sub_cats = os.listdir(f'{samples_path}/{i}')
            for j in sub_cats:
                if '.root' in os.listdir(f'{samples_path}/{i}/{j}')[0]:
                    files = os.listdir(f'{samples_path}/{i}/{j}')
                    filelist[f'{i}_{j}'] = [f'{samples_path}/{i}/{j}/{file}' for file in files]

    # Convert to input dictionary format
    input_dict = {}
    for i in filelist:
        input_dict[i] = {}
        input_dict[i]['files'] = {}
        for j in filelist[i]:
            input_dict[i]['files'][j] = {'object_path': 'Events'}

    samples = input_dict

    print('Preprocessing samples...')
    
    @dask.delayed
    def sampler(samples):
        samples_ready, samples = dataset_tools.preprocess(
            samples,
            step_size=step_size,
            skip_bad_files=True,
            recalculate_steps=True,
            save_form=False,
        )
        return samples_ready

    sampler_dict = {}
    for i in samples:
        sampler_dict[i] = sampler(samples[i])

    print('Computing preprocessing tasks...')
    if manager:
        samples_postprocess = dask.compute(
            sampler_dict,
            scheduler=manager.get,
            progress_disable=True,
            resources={"cores": 1},
            resources_mode=None,
            worker_transfers=True,
        )[0]
    else:
        samples_postprocess = dask.compute(sampler_dict)[0]

    samples_ready = {}
    for i in samples_postprocess:
        samples_ready[i] = samples_postprocess[i]['files']

    print('Preprocessing complete!')
    
    return samples_ready


def filter_existing_files(samples_ready):
    """Filter samples to only include files that exist on disk."""
    filtered_samples = {}
    for dataset, info in samples_ready.items():
        files = info.get("files", {})
        existing_files = {
            path: meta for path, meta in files.items() if os.path.exists(path)
        }
        if existing_files:
            filtered_samples[dataset] = {
                "files": existing_files,
                "form": info["form"],
                "metadata": info["metadata"],
            }
    return filtered_samples


def show_available_samples(samples_ready):
    """Print available samples and their file counts."""
    print("\nAvailable samples and their file counts:")
    print("-" * 40)
    for key, value in samples_ready.items():
        print(f"{key}: {len(value['files'])} files")
    print("-" * 40)


def analysis(events, ecf_upper_bound=3, triggers_file='triggers.json'):
    """
    Main analysis function that processes events and calculates ECFs.
    
    Parameters:
    -----------
    events : awkward array
        Input events from NanoAOD
    ecf_upper_bound : int
        Upper bound for ECF n-point calculations (default: 3)
    triggers_file : str
        Path to triggers JSON file
        
    Returns:
    --------
    dask_awkward array : Delayed task for writing parquet output
    """
    from variable_functions import color_ring
    
    dataset = events.metadata["dataset"]
    print(f"Processing dataset: {dataset}")

    # Apply puppi weights to particle candidates
    events['PFCands', 'pt'] = (
        events.PFCands.pt * events.PFCands.puppiWeight
    )
    
    # Fix for softdrop issue
    cut_to_fix_softdrop = (ak.num(events.FatJet.constituents.pf, axis=2) > 0)
    events = events[ak.all(cut_to_fix_softdrop, axis=1)]
    
    # Load triggers
    with open(triggers_file, 'r') as f:
        triggers = json.load(f)

    trigger = ak.zeros_like(ak.firsts(events.FatJet.pt), dtype='bool')
    for t in triggers['2017']:
        if t in events.HLT.fields:
            trigger = trigger | events.HLT[t]
    trigger = ak.fill_none(trigger, False)

    events['FatJet', 'num_fatjets'] = ak.num(events.FatJet)

    # Muon selection
    goodmuon = (
        (events.Muon.pt > 10)
        & (abs(events.Muon.eta) < 2.4)
        & (events.Muon.pfRelIso04_all < 0.25)
        & events.Muon.looseId
    )
    nmuons = ak.sum(goodmuon, axis=1)
    leadingmuon = ak.firsts(events.Muon[goodmuon])

    # Electron selection
    goodelectron = (
        (events.Electron.pt > 10)
        & (abs(events.Electron.eta) < 2.5)
        & (events.Electron.cutBased >= 2)
    )
    nelectrons = ak.sum(goodelectron, axis=1)

    # Tau selection
    ntaus = ak.sum(
        (
            (events.Tau.pt > 20)
            & (abs(events.Tau.eta) < 2.3)
            & (events.Tau.rawIso < 5)
            & (events.Tau.idDeepTau2017v2p1VSjet)
            & ak.all(events.Tau.metric_table(events.Muon[goodmuon]) > 0.4, axis=2)
            & ak.all(events.Tau.metric_table(events.Electron[goodelectron]) > 0.4, axis=2)
        ),
        axis=1,
    )

    # Region selection (one muon)
    onemuon = ((nmuons == 1) & (nelectrons == 0) & (ntaus == 0))
    region = onemuon

    # B-tag counting
    events['btag_count'] = ak.sum(
        events.Jet[(events.Jet.pt > 20) & (abs(events.Jet.eta) < 2.4)].btagDeepFlavB > 0.3040, 
        axis=1
    )

    # Gen-matching for signal/background
    if ('hgg' in dataset) or ('hbb' in dataset):
        print("Signal: Higgs jets")
        genhiggs = events.GenPart[
            (events.GenPart.pdgId == 25)
            & events.GenPart.hasFlags(["fromHardProcess", "isLastCopy"])
        ]
        parents = events.FatJet.nearest(genhiggs, threshold=0.2)
        higgs_jets = ~ak.is_none(parents, axis=1)
        events['GenMatch_Mask'] = higgs_jets

    elif ('wqq' in dataset) or ('ww' in dataset):
        print('Background: W jets')
        genw = events.GenPart[
            (abs(events.GenPart.pdgId) == 24)
            & events.GenPart.hasFlags(['fromHardProcess', 'isLastCopy'])
        ]
        parents = events.FatJet.nearest(genw, threshold=0.2)
        w_jets = ~ak.is_none(parents, axis=1)
        events['GenMatch_Mask'] = w_jets

    elif ('zqq' in dataset) or ('zz' in dataset):
        print('Background: Z jets')
        genz = events.GenPart[
            (events.GenPart.pdgId == 23)
            & events.GenPart.hasFlags(['fromHardProcess', 'isLastCopy'])
        ]
        parents = events.FatJet.nearest(genz, threshold=0.2)
        z_jets = ~ak.is_none(parents, axis=1)
        events['GenMatch_Mask'] = z_jets

    elif ('wz' in dataset):
        print('Background: WZ jets')
        genwz = events.GenPart[
            ((abs(events.GenPart.pdgId) == 24)|(events.GenPart.pdgId == 23))
            & events.GenPart.hasFlags(["fromHardProcess", "isLastCopy"])
        ]
        parents = events.FatJet.nearest(genwz, threshold=0.2)
        wz_jets = ~ak.is_none(parents, axis=1)
        events['GenMatch_Mask'] = wz_jets

    # Fat jet selection
    fatjetSelect = (
        (events.FatJet.pt > 400)
        & (abs(events.FatJet.eta) < 2.4)
        & (events.FatJet.msoftdrop > 40)
        & (events.FatJet.msoftdrop < 200)
        & (region)
        & (trigger)
    )
    
    events["goodjets"] = events.FatJet[fatjetSelect]
    mask = ~ak.is_none(ak.firsts(events.goodjets))
    events = events[mask]
    
    # Calculate color ring
    events['goodjets', 'color_ring'] = ak.unflatten(
        color_ring(events.goodjets, cluster_val=0.4), 
        counts=ak.num(events.goodjets)
    )

    # Calculate ECFs
    jetdef = fastjet.JetDefinition(fastjet.cambridge_algorithm, 0.8)
    pf = ak.flatten(events.goodjets.constituents.pf, axis=1)
    cluster = fastjet.ClusterSequence(pf, jetdef)
    softdrop = cluster.exclusive_jets_softdrop_grooming()
    softdrop_cluster = fastjet.ClusterSequence(softdrop.constituents, jetdef)
    
    ecfs = {}
    for n in range(2, ecf_upper_bound + 1):
        for v in range(1, int(scipy.special.binom(n, 2)) + 1):
            for b in range(5, 45, 5):
                ecf_name = f'{v}e{n}^{b/10}'
                ecfs[ecf_name] = ak.unflatten(
                    softdrop_cluster.exclusive_jets_energy_correlator(
                        func='generic', npoint=n, angles=v, beta=b/10
                    ), 
                    counts=ak.num(events.goodjets)
                )
    events["ecfs"] = ak.zip(ecfs)

    # Create output structure
    skim_fields = {
        "Color_Ring": events.goodjets.color_ring,
        "ECFs": events.ecfs,
        "msoftdrop": events.goodjets.msoftdrop,
        "pt": events.goodjets.pt,
        "btag_ak4s": events.btag_count,
        "pn_HbbvsQCD": events.goodjets.particleNet_HbbvsQCD,
        "pn_md": events.goodjets.particleNetMD_QCD,
    }
    
    # Add matching field for signal/background samples
    if ('hgg' in dataset) or ('hbb' in dataset) or ('wqq' in dataset) or \
       ('ww' in dataset) or ('zqq' in dataset) or ('zz' in dataset) or ('wz' in dataset):
        skim_fields["matching"] = events.GenMatch_Mask
    
    skim = ak.zip(skim_fields, depth_limit=1)
       
    output_path = f"output/{dataset}/"
    os.makedirs(output_path, exist_ok=True)
    
    skim_task = dak.to_parquet(skim, output_path, compute=False)
    return skim_task
