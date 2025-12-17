from utils.kinematics import *
from utils.selection import *
from utils.corrections import *

from coffea import processor, lookup_tools
from coffea.util import save

import os, correctionlib, json

scriptPath = os.path.dirname(os.path.abspath(__file__))

class my_processor(processor.ProcessorABC):
    def __init__(self, year='2016preVFP', type='data'):
        self.variables_to_store = [
            # channels
            'em_ch', 'eloosem_ch', 'looseem_ch', 'looseeloosem_ch',
            'et_ch', 'elooset_ch', 'looseet_ch', 'looseelooset_ch',
            'mt_ch', 'mlooset_ch', 'loosemt_ch', 'loosemlooset_ch',

            # lepton kinematics
            'L1_lab_pt', 'L1_H_pt', 'L1_H_p', 'L1_met_mT', 'L1_met_DeltaPhi', 'col_mass_L1', 'L1_M', 'L1_eta', 'L1_genPartFlav',
            'L2_lab_pt', 'L2_H_pt', 'L2_H_p', 'L2_met_mT', 'L2_met_DeltaPhi', 'col_mass_L2', 'L2_M', 'L2_eta', 'L2_genPartFlav',
            'L1_L2_DeltaEta', 'L1_L2_DeltaPhi', 'dilep_mass',

            # jet kinematics
            'J1_lab_pt', 'J1_eta', 'J1_phi', 'J1_mass',
            'J2_lab_pt', 'J2_eta', 'J2_phi', 'J2_mass', 
            'nJets', 'Mjj', 'J1_J2_DeltaEta',
            'GenJet_pt', 'GenJet_eta', 'GenJet_phi', 'GenJet_mass', 'GenJetIdx',

            # miscellaneous
            'met', 'mtrigger', 'etrigger', 'passVBFcut', 'Tau_DM', 'OppCharge', 'delta_R', 'bTagSF', 'weight' # 'third_lepton_veto'
        ]
        self.accumulator = {}
        self._year = year
        self._type = type
        rochester_data = lookup_tools.txt_converters.convert_rochester_file(f"{scriptPath}/../correction_files/roccor/RoccoR{self._year}UL.txt", loaduncs=True)
        with open(f"{scriptPath}/../correction_files/goldenJSON/golden_{self._year}.json", 'r') as j: 
            goldenJSON = json.load(j)
        self.correction_files = {
            'goldenJSON' : goldenJSON,
            'ele_sf'     : correctionlib.CorrectionSet.from_file(f"/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/EGM/{self._year}_UL/electron.json.gz"),
            'ele_sf_pri' : correctionlib.CorrectionSet.from_file(f"{scriptPath}/../correction_files/electron_files/{self._year}_UL/electron_hem.json"),
            'muon_sf'    : correctionlib.CorrectionSet.from_file(f"/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/MUO/{self._year}_UL/muon_Z.json.gz"),
            'btag_sf'    : correctionlib.CorrectionSet.from_file(f"/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/{self._year}_UL/btagging.json.gz"),
            'tau_sf'     : correctionlib.CorrectionSet.from_file(f"/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/TAU/{self._year}_UL/tau.json.gz"),
            'pu_sf'      : correctionlib.CorrectionSet.from_file(f"/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/LUM/{self._year}_UL/puWeights.json.gz"),
            'jet_jerc'   : correctionlib.CorrectionSet.from_file(f"/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/{self._year}_UL/jet_jerc.json.gz"),
            'jer_tool'   : correctionlib.CorrectionSet.from_file(f"/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/jer_smear.json.gz"),
            'met_tool'   : correctionlib.CorrectionSet.from_file(f"/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/{self._year}_UL/met.json.gz"),
            'jetvetomap' : correctionlib.CorrectionSet.from_file(f"/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/{self._year}_UL/jetvetomaps.json.gz"),
            'rochester'  : lookup_tools.rochester_lookup.rochester_lookup(rochester_data)
        } # for help, use `correction summary <name>.json.gz` in commandline

    def process(self, events):
        yr, tp, corr = self._year, self._type, self.correction_files

        # Energy Correction (corrections.py)
        muRochesterCorr(events, tp, corr['rochester'])
        tauCorr(events, tp, corr['tau_sf'])
        jetCorr(events, yr, tp, corr['jet_jerc'], corr['jer_tool'])
        metCorr(events, tp, corr['met_tool'])

        # Lepton Selection (selection.py)
        DefineLeptons(events)
        DefineChannels(events)
        CollectLeptons(events, tp)

        # Event selection (selection.py)
        if tp=='data': events = SelectGoldenLumiEvents(events, corr['goldenJSON'])
        events = ThirdLeptonVeto(events)
        events = SelectDilepEvents(events)
        events = SelectTrigMatchedEvents(events, yr)

        # Jets (selection.py)
        JetCleaning(events, yr)
        events = JetVeto(events, yr, tp, corr['jetvetomap'])
        events = ak.drop_none(events)

        # Scale factor (corrections.py)
        if len(events)>0:
            evaluate_bareWeight(events, tp)
            evaluate_btagSF(events, tp, corr['btag_sf'], 'L')
            if tp=='mc':
                evaluate_prefireWeight(events, yr, tp)
                evaluate_pileupWeight(events, yr, tp, corr['pu_sf'])
                evaluate_ElectronSF(events, yr, tp, corr['ele_sf'], corr['ele_sf_pri'])
                evaluate_MuonSF(events, yr, tp, corr['muon_sf'])
                evaluate_TauSF(events, tp, corr['tau_sf'])
            kinematics(events, tp)
            for variable in self.variables_to_store:
                self.accumulator[variable, events.metadata["dataset"]] = events[variable].to_list()
        else:
            print(f'No events in {events.metadata["dataset"]}!')

        return self.accumulator
        
    def postprocess(self, accumulator):
        pass