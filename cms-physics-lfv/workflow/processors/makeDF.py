from utils.kinematics import *
from utils.selection import *


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
            'met', 'mtrigger', 'etrigger', 'passVBFcut', 'Tau_DM', 'OppCharge', 'delta_R',  'weight'
        ]
        self.accumulator = {}
        self._year = year
        self._type = type

    def process(self, events):
        yr, tp = self._year, self._type

        # Energy Correction (corrections.py)

        # Lepton Selection (selection.py)
        DefineLeptons(events)
        DefineChannels(events)
        CollectLeptons(events, tp)

        # Event selection (selection.py)

        events = ThirdLeptonVeto(events)
        events = SelectDilepEvents(events)
        events = SelectTrigMatchedEvents(events, yr)

        # Jets (selection.py)
        JetCleaning(events, yr)
        events = ak.drop_none(events)

        # Scale factor (corrections.py)
        if len(events)>0:
            evaluate_bareWeight(events, tp)
            kinematics(events, tp)
            for variable in self.variables_to_store:
                self.accumulator[variable, events.metadata["dataset"]] = events[variable].to_list()
        else:
            print(f'No events in {events.metadata["dataset"]}!')

        return self.accumulator
        
    def postprocess(self, accumulator):
        pass
