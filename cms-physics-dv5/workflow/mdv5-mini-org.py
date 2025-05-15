#! /usr/bin/env python

from coffea import dataset_tools
from coffea.nanoevents import PFNanoAODSchema

import awkward as ak
import dask
import fastjet
import json
import ndcctools.taskvine as vine
import os
import scipy
import warnings

warnings.filterwarnings("ignore", "Found duplicate branch")
warnings.filterwarnings("ignore", "Missing cross-reference index for")
warnings.filterwarnings("ignore", "dcut")
warnings.filterwarnings("ignore", "Please ensure")
warnings.filterwarnings("ignore", "invalid value")
warnings.filterwarnings("ignore", module="coffea.*")


samples_dict_file = "data/samples_ready.json"
triggers_file = "data/triggers.json"

with open(samples_dict_file) as f:
    samples_dict_rel = json.load(f)
    samples_dict = {}
    for ds_name, ds in samples_dict_rel.items():
        samples_dict[ds_name] = {"files": {}}

        for filename, file_info in ds["files"].items():
            samples_dict[ds_name]["files"][os.path.abspath(filename)] = file_info


with open(triggers_file) as f:
    triggers = json.load(f)


def analysis(events):
    warnings.filterwarnings("ignore", module="coffea.*")

    dataset = events.metadata["dataset"]
    events["PFCands", "pt"] = events.PFCands.pt * events.PFCands.puppiWeight

    cut_to_fix_softdrop = ak.num(events.FatJet.constituents.pf, axis=2) > 0
    events = events[ak.all(cut_to_fix_softdrop, axis=1)]

    trigger = ak.zeros_like(ak.firsts(events.FatJet.pt), dtype="bool")
    for t in triggers["2017"]:
        if t in events.HLT.fields:
            trigger = trigger | events.HLT[t]
    trigger = ak.fill_none(trigger, False)

    events["FatJet", "num_fatjets"] = ak.num(events.FatJet)

    goodmuon = (
        (events.Muon.pt > 10)
        & (abs(events.Muon.eta) < 2.4)
        & (events.Muon.pfRelIso04_all < 0.25)
        & events.Muon.looseId
    )

    nmuons = ak.sum(goodmuon, axis=1)

    goodelectron = (
        (events.Electron.pt > 10)
        & (abs(events.Electron.eta) < 2.5)
        & (events.Electron.cutBased >= 2)  # events.Electron.LOOSE
    )
    nelectrons = ak.sum(goodelectron, axis=1)

    ntaus = ak.sum(
        (
            (events.Tau.pt > 20)
            & (abs(events.Tau.eta) < 2.3)
            & (events.Tau.rawIso < 5)
            & (events.Tau.idDeepTau2017v2p1VSjet)
            & ak.all(events.Tau.metric_table(events.Muon[goodmuon]) > 0.4, axis=2)
            & ak.all(
                events.Tau.metric_table(events.Electron[goodelectron]) > 0.4, axis=2
            )
        ),
        axis=1,
    )

    nolepton = (nmuons == 0) & (nelectrons == 0) & (ntaus == 0)
    onemuon = (nmuons == 1) & (nelectrons == 0) & (ntaus == 0)

    region = onemuon  # Use this option to let less data through the cuts
    region = nolepton  # Use this option to let more data through the cuts

    events["btag_count"] = ak.sum(
        events.Jet[(events.Jet.pt > 20) & (abs(events.Jet.eta) < 2.4)].btagDeepFlavB
        > 0.3040,
        axis=1,
    )

    if ("hgg" in dataset) or ("hbb" in dataset):
        print("signal")
        genhiggs = events.GenPart[
            (events.GenPart.pdgId == 25)
            & events.GenPart.hasFlags(["fromHardProcess", "isLastCopy"])
        ]
        parents = events.FatJet.nearest(genhiggs, threshold=0.2)
        higgs_jets = ~ak.is_none(parents, axis=1)
        events["GenMatch_Mask"] = higgs_jets

        fatjetSelect = (
            (events.FatJet.pt > 400)
            # & (events.FatJet.pt < 1200)
            & (abs(events.FatJet.eta) < 2.4)
            & (events.FatJet.msoftdrop > 40)
            & (events.FatJet.msoftdrop < 200)
            & (region)
            & (trigger)
        )

    elif ("wqq" in dataset) or ("ww" in dataset):
        print("w background")
        genw = events.GenPart[
            (abs(events.GenPart.pdgId) == 24)
            & events.GenPart.hasFlags(["fromHardProcess", "isLastCopy"])
        ]
        parents = events.FatJet.nearest(genw, threshold=0.2)
        w_jets = ~ak.is_none(parents, axis=1)
        events["GenMatch_Mask"] = w_jets

        fatjetSelect = (
            (events.FatJet.pt > 400)
            # & (events.FatJet.pt < 1200)
            & (abs(events.FatJet.eta) < 2.4)
            & (events.FatJet.msoftdrop > 40)
            & (events.FatJet.msoftdrop < 200)
            & (region)
            & (trigger)
        )

    elif ("zqq" in dataset) or ("zz" in dataset):
        print("z background")
        genz = events.GenPart[
            (events.GenPart.pdgId == 23)
            & events.GenPart.hasFlags(["fromHardProcess", "isLastCopy"])
        ]
        parents = events.FatJet.nearest(genz, threshold=0.2)
        z_jets = ~ak.is_none(parents, axis=1)
        events["GenMatch_Mask"] = z_jets

        fatjetSelect = (
            (events.FatJet.pt > 400)
            # & (events.FatJet.pt < 1200)
            & (abs(events.FatJet.eta) < 2.4)
            & (events.FatJet.msoftdrop > 40)
            & (events.FatJet.msoftdrop < 200)
            & (region)
            & (trigger)
        )

    elif "wz" in dataset:
        print("wz background")
        genwz = events.GenPart[
            ((abs(events.GenPart.pdgId) == 24) | (events.GenPart.pdgId == 23))
            & events.GenPart.hasFlags(["fromHardProcess", "isLastCopy"])
        ]
        parents = events.FatJet.nearest(genwz, threshold=0.2)
        wz_jets = ~ak.is_none(parents, axis=1)
        events["GenMatch_Mask"] = wz_jets

        fatjetSelect = (
            (events.FatJet.pt > 400)
            # & (events.FatJet.pt < 1200)
            & (abs(events.FatJet.eta) < 2.4)
            & (events.FatJet.msoftdrop > 40)
            & (events.FatJet.msoftdrop < 200)
            & (region)
            & (trigger)
        )

    else:
        print("background")
        fatjetSelect = (
            (events.FatJet.pt > 400)
            # & (events.FatJet.pt < 1200)
            & (abs(events.FatJet.eta) < 2.4)
            & (events.FatJet.msoftdrop > 40)
            & (events.FatJet.msoftdrop < 200)
            & (region)
            & (trigger)
        )

    events["goodjets"] = events.FatJet[fatjetSelect]
    mask = ~ak.is_none(ak.firsts(events.goodjets))
    events = events[mask]
    ecfs = {}

    # events['goodjets', 'color_ring'] = ak.unflatten(
    #         color_ring(events.goodjets, cluster_val=0.4), counts=ak.num(events.goodjets)
    # )

    jetdef = fastjet.JetDefinition(fastjet.cambridge_algorithm, 0.8)
    pf = ak.flatten(events.goodjets.constituents.pf, axis=1)
    cluster = fastjet.ClusterSequence(pf, jetdef)
    softdrop = cluster.exclusive_jets_softdrop_grooming()
    softdrop_cluster = fastjet.ClusterSequence(softdrop.constituents, jetdef)

    upper_bound = 6
    upper_bound = 3
    for n in range(2, upper_bound):
        for v in range(1, int(scipy.special.binom(n, 2)) + 1):
            for b in range(5, 45, 5):
                ecf_name = f"{v}e{n}^{b/10}"
                ecfs[ecf_name] = ak.unflatten(
                    softdrop_cluster.exclusive_jets_energy_correlator(
                        func="generic", npoint=n, angles=v, beta=b / 10
                    ),
                    counts=ak.num(events.goodjets),
                )
    events["ecfs"] = ak.zip(ecfs)

    if (
        ("hgg" in dataset)
        or ("hbb" in dataset)
        or ("wqq" in dataset)
        or ("ww" in dataset)
        or ("zqq" in dataset)
        or ("zz" in dataset)
        or ("wz" in dataset)
    ):
        skim = ak.zip(
            {
                # "Color_Ring": events.goodjets.color_ring,
                "ECFs": events.ecfs,
                "msoftdrop": events.goodjets.msoftdrop,
                "pt": events.goodjets.pt,
                "btag_ak4s": events.btag_count,
                "pn_HbbvsQCD": events.goodjets.particleNet_HbbvsQCD,
                "pn_md": events.goodjets.particleNetMD_QCD,
                "matching": events.GenMatch_Mask,
            },
            depth_limit=1,
        )
    else:
        skim = ak.zip(
            {
                # "Color_Ring": events.goodjets.color_ring,
                "ECFs": events.ecfs,
                "msoftdrop": events.goodjets.msoftdrop,
                "pt": events.goodjets.pt,
                "btag_ak4s": events.btag_count,
                "pn_HbbvsQCD": events.goodjets.particleNet_HbbvsQCD,
                "pn_md": events.goodjets.particleNetMD_QCD,
            },
            depth_limit=1,
        )

    # skim_task = dak.to_parquet(
    #     # events,
    #     skim,
    #     f"/scratch365/btovar/ecf_calculator_output/{dataset}/"
    #     compute=False,
    # )
    # return skim_task
    return skim


tasks = dataset_tools.apply_to_fileset(
    analysis,
    samples_dict,
    uproot_options={},
    schemaclass=PFNanoAODSchema,
)

m = vine.DaskVine(
    [9123, 9128],
    name=f"{os.environ['USER']}-mini-mdv5",
)

tasks = dataset_tools.apply_to_fileset(
    analysis,
    samples_dict,
    uproot_options={},
    schemaclass=PFNanoAODSchema,
)

m = vine.DaskVine(
    [9123, 9128],
    name=f"{os.environ['USER']}-mini-mdv5",
)

computed = dask.compute(
    tasks,
    scheduler=m.get,
    resources={"cores": 1},
    resources_mode=None,
)

print(computed)
