import awkward as ak
from coffea.nanoevents.methods import candidate
import numpy as np

def DefineLeptons(events):
    # Muon selection
    events["Muon", "base"]      = (events.Muon.pt > 15) & (abs(events.Muon.eta) < 2.4) & (abs(events.Muon.dxy) < 0.2) & (abs(events.Muon.dz) < 0.5)
    events["Muon", "veto"]      = (events.Muon.base) & (events.Muon.looseId) & (events.Muon.pfRelIso04_all < 0.5)
    events["Muon", "TauH"]      = (events.Muon.base) & (events.Muon.tightId) & (events.Muon.pfRelIso04_all < 0.15)
    events["Muon", "TauL"]      = (events.Muon.base) & (events.Muon.tightId) & (events.Muon.pfRelIso04_all < 0.15)
    events["Muon", "looseTauH"] = (events.Muon.base) & (events.Muon.looseId) & ~(events.Muon.tightId) & (events.Muon.pfRelIso04_all < 0.5)
    events["Muon", "looseTauL"] = (events.Muon.base) & (events.Muon.looseId) & ~(events.Muon.tightId) & (events.Muon.pfRelIso04_all < 0.5)

    # Electron selection
    events["Electron", "base"]      = (events.Electron.pt > 20) & (abs(events.Electron.eta) < 2.5) & (abs(events.Electron.dxy) < 0.2) & (abs(events.Electron.dz) < 0.5) &  ~((abs(events.Electron.eta) < 1.566) & (abs(events.Electron.eta) > 1.442))
    events["Electron", "veto"]      = (events.Electron.base) & (events.Electron.mvaFall17V2noIso_WPL)  & (events.Electron.pfRelIso03_all < 0.5)
    events["Electron", "TauH"]      = (events.Electron.base) & (events.Electron.mvaFall17V2noIso_WP80) & (events.Electron.pfRelIso03_all < 0.1)
    events["Electron", "TauL"]      = (events.Electron.base) & (events.Electron.mvaFall17V2noIso_WP80) & (events.Electron.pfRelIso03_all < 0.1)
    events["Electron", "looseTauH"] = (events.Electron.base) & (events.Electron.mvaFall17V2noIso_WPL)  & ~(events.Electron.mvaFall17V2noIso_WP80) & (events.Electron.pfRelIso03_all < 0.5)
    events["Electron", "looseTauL"] = (events.Electron.base) & (events.Electron.mvaFall17V2noIso_WPL)  & ~(events.Electron.mvaFall17V2noIso_WP80) & (events.Electron.pfRelIso03_all < 0.5)

    # Tau selection
    events["Tau", "base"]      = (events.Tau.pt > 30) & (abs(events.Tau.eta) < 2.3) & (abs(events.Tau.dz) < 0.2) & (events.Tau.idDeepTau2017v2p1VSe >= 2) & (events.Tau.idDeepTau2017v2p1VSmu >= 8) & (events.Tau.decayMode!=5) & (events.Tau.decayMode!=6)
    events["Tau", "veto"]      = (events.Tau.base) & (events.Tau.idDeepTau2017v2p1VSjet >= 16)
    events["Tau", "TauH"]      = (events.Tau.base) & (events.Tau.idDeepTau2017v2p1VSe >= 32) & (events.Tau.idDeepTau2017v2p1VSjet >= 32)
    events["Tau", "looseTauH"] = (events.Tau.base) & (events.Tau.idDeepTau2017v2p1VSe >= 32) & (events.Tau.idDeepTau2017v2p1VSjet >= 8) & (events.Tau.idDeepTau2017v2p1VSjet < 32)

def DefineChannels(events):
    # em channel
    events['em_ch']           = (ak.count_nonzero(events.Muon.TauL, -1)==1)      & (ak.count_nonzero(events.Electron.TauL, -1)==1)
    events['eloosem_ch']      = (ak.count_nonzero(events.Muon.looseTauL, -1)==1) & (ak.count_nonzero(events.Electron.TauL, -1)==1)
    events['looseem_ch']      = (ak.count_nonzero(events.Muon.TauL, -1)==1)      & (ak.count_nonzero(events.Electron.looseTauL, -1)==1)
    events['looseeloosem_ch'] = (ak.count_nonzero(events.Muon.looseTauL, -1)==1) & (ak.count_nonzero(events.Electron.looseTauL, -1)==1)
    
    # et channel
    events['et_ch']           = (ak.count_nonzero(events.Electron.TauH, -1)==1)      & (ak.count_nonzero(events.Tau.TauH, -1)==1)
    events['elooset_ch']      = (ak.count_nonzero(events.Electron.TauH, -1)==1)      & (ak.count_nonzero(events.Tau.looseTauH, -1)==1)
    events['looseet_ch']      = (ak.count_nonzero(events.Electron.looseTauH, -1)==1) & (ak.count_nonzero(events.Tau.TauH, -1)==1)
    events['looseelooset_ch'] = (ak.count_nonzero(events.Electron.looseTauH, -1)==1) & (ak.count_nonzero(events.Tau.looseTauH, -1)==1)

    # mt channel
    events['mt_ch']           = (ak.count_nonzero(events.Muon.TauH, -1)==1)      & (ak.count_nonzero(events.Tau.TauH, -1)==1)
    events['mlooset_ch']      = (ak.count_nonzero(events.Muon.TauH, -1)==1)      & (ak.count_nonzero(events.Tau.looseTauH, -1)==1)
    events['loosemt_ch']      = (ak.count_nonzero(events.Muon.looseTauH, -1)==1) & (ak.count_nonzero(events.Tau.TauH, -1)==1)
    events['loosemlooset_ch'] = (ak.count_nonzero(events.Muon.looseTauH, -1)==1) & (ak.count_nonzero(events.Tau.looseTauH, -1)==1)

def CollectLeptons(events, type):
    # L1 is the lighter lepton by convention
    L1_collections = ak.where(events.em_ch      | events.eloosem_ch     , events.Electron[events.Electron.TauL]     , events.Electron[events.Electron.TauH])
    L1_collections = ak.where(events.et_ch      | events.elooset_ch     , events.Electron[events.Electron.TauH]     , L1_collections)
    L1_collections = ak.where(events.looseem_ch | events.looseeloosem_ch, events.Electron[events.Electron.looseTauL], L1_collections)
    L1_collections = ak.where(events.looseet_ch | events.looseelooset_ch, events.Electron[events.Electron.looseTauH], L1_collections)
    L1_collections = ak.where(events.mt_ch      | events.mlooset_ch     , events.Muon[events.Muon.TauH]             , L1_collections)
    L1_collections = ak.where(events.loosemt_ch | events.loosemlooset_ch, events.Muon[events.Muon.looseTauH]        , L1_collections)
    
    L2_collections = ak.where(events.em_ch      | events.looseem_ch     , events.Muon[events.Muon.TauL]     , events.Muon[events.Muon.TauH])
    L2_collections = ak.where(events.eloosem_ch | events.looseeloosem_ch, events.Muon[events.Muon.looseTauL], L2_collections) 
    L2_collections = ak.where(events.et_ch      | events.looseet_ch     , events.Tau[events.Tau.TauH]       , L2_collections)
    L2_collections = ak.where(events.mt_ch      | events.loosemt_ch     , events.Tau[events.Tau.TauH]       , L2_collections)
    L2_collections = ak.where(events.elooset_ch | events.looseelooset_ch, events.Tau[events.Tau.looseTauH]  , L2_collections)
    L2_collections = ak.where(events.mlooset_ch | events.loosemlooset_ch, events.Tau[events.Tau.looseTauH]  , L2_collections)

    L1_collections = ak.pad_none(L1_collections, 1, axis=-1)[:,0]
    L2_collections = ak.pad_none(L2_collections, 1, axis=-1)[:,0]

    events['L1_collections'] = ak.zip(
        {
            "pt"    : L1_collections.pt,
            "eta"   : L1_collections.eta,
            "phi"   : L1_collections.phi,
            "mass"  : L1_collections.mass,
            "charge": L1_collections.charge,
            "genPartFlav": L1_collections.genPartFlav if type=='mc' else ak.zeros_like(L1_collections.pt)
        }, 
        with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)
    
    events['L2_collections'] = ak.zip(
        {
            "pt"    : L2_collections.pt,
            "eta"   : L2_collections.eta,
            "phi"   : L2_collections.phi,
            "mass"  : L2_collections.mass,
            "charge": L2_collections.charge,
            "genPartFlav": L2_collections.genPartFlav if type=='mc' else ak.zeros_like(L2_collections.pt)
        },
        with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)

    events['M_collections'] = ak.pad_none(events.Muon[(events.Muon.TauH | events.Muon.looseTauH | events.Muon.TauL | events.Muon.looseTauL)], 1, axis=-1)[:,0]
    events['E_collections'] = ak.pad_none(events.Electron[(events.Electron.TauH | events.Electron.looseTauH | events.Electron.TauL | events.Electron.looseTauL)], 1, axis=-1)[:,0]
    events['T_collections'] = ak.pad_none(events.Tau[events.Tau.TauH | events.Tau.looseTauH], 1, axis=-1)[:,0]

def SelectGoldenLumiEvents(events, goldenJSON):
    goldenCut = ak.Array([False] * len(events))
    for run, lumis in goldenJSON.items():
        run_match = events.run==int(run)
        good_lumis = np.concatenate([np.arange(a, b + 1) for a, b in lumis])
        lum_match = np.isin(events.luminosityBlock, good_lumis)
        goldenCut = ak.where(run_match & lum_match, True, goldenCut)
    return events[goldenCut]

def ThirdLeptonVeto(events):
    third_lepton_veto_cut = (ak.count_nonzero(events.Muon.veto, -1) + ak.count_nonzero(events.Electron.veto, -1) + ak.count_nonzero(events.Tau.veto, -1))==2
    return events[third_lepton_veto_cut]

def SelectDilepEvents(events):
    events["OppCharge"] = (ak.fill_none(events.M_collections.charge, 1) * ak.fill_none(events.E_collections.charge, 1) * ak.fill_none(events.T_collections.charge, 1)) == -1
    events["delta_R"]   = (events.M_collections.delta_r(events.E_collections)) 
    OrthoCut = ((0+
        events.em_ch + events.eloosem_ch + events.looseem_ch + events.looseeloosem_ch +
        events.et_ch + events.elooset_ch + events.looseet_ch + events.looseelooset_ch +
        events.mt_ch + events.mlooset_ch + events.loosemt_ch + events.loosemlooset_ch
    )==1)
    DiLep_dr0p4 = ak.fill_none(events['L1_collections'].delta_r(events['L2_collections']) > 0.4, False, axis=-1)
    dilepcut = OrthoCut & DiLep_dr0p4
    return events[dilepcut]

def SelectTrigMatchedEvents(events, year):
    if year=='2016preVFP' or year=='2016postVFP':
        ept = 27
        mpt = 26
        events['mtrigger'] = events.HLT.IsoMu24 | events.HLT.IsoTkMu24
        events['etrigger'] = events.HLT.Ele27_WPTight_Gsf

    elif year=='2017':
        ept = 34
        mpt = 29
        events['mtrigger'] = events.HLT.IsoMu27
        events['etrigger'] = events.HLT.Ele32_WPTight_Gsf_L1DoubleEG
    
    elif year=='2018':
        ept = 34
        mpt = 26
        events['mtrigger'] = events.HLT.IsoMu24
        events['etrigger'] = events.HLT.Ele32_WPTight_Gsf

    events['mtrigger'] = ak.fill_none(events.M_collections.pt > mpt, False) & events.mtrigger
    events['etrigger'] = ak.fill_none(events.E_collections.pt > ept, False) & events.etrigger
    
    # TODO: This part is newly added! Verify with Reyer.
    etrg_collections = events.TrigObj[(events.TrigObj.id == 11) & (events.TrigObj.pt > ept) & (events.TrigObj.filterBits & 2 == 2)]
    mtrg_collections = events.TrigObj[(events.TrigObj.id == 13) & (events.TrigObj.pt > mpt) & (events.TrigObj.filterBits & 2 == 2)]

    etrg_Match = ak.fill_none((ak.any(events.E_collections.delta_r(etrg_collections) < 0.3, axis=1)) & (events.E_collections.pt > ept), False)
    mtrg_Match = ak.fill_none((ak.any(events.M_collections.delta_r(mtrg_collections) < 0.3, axis=1)) & (events.M_collections.pt > mpt), False)
    TrigCut = (etrg_Match & events.etrigger) | (mtrg_Match & events.mtrigger)

    return events[TrigCut]

def JetCleaning(events, year):
    btag_WP = {
         "L": {'2016preVFP': 0.0508, '2016postVFP': 0.0480, '2017': 0.0532, '2018': 0.0490},
         "M": {'2016preVFP': 0.2598, '2016postVFP': 0.2489, '2017': 0.3040, '2018': 0.2783},
         "T": {'2016preVFP': 0.6502, '2016postVFP': 0.6377, '2017': 0.7476, '2018': 0.7100},
    }

    if '2016' in year:
        events['Jet', 'passJet30ID'] = ak.fill_none(events.Jet.delta_r(events.L1_collections) > 0.4, True) & ak.fill_none(events.Jet.delta_r(events.L2_collections) > 0.4, True) & (events.Jet['pt']>30) & ((events.Jet.jetId>>1) & 1) & (abs(events.Jet.eta)<4.7) & ((events.Jet.puId&1) | (events.Jet['pt']>50)) 
    else:
        events['Jet', 'passJet30ID'] = ak.fill_none(events.Jet.delta_r(events.L1_collections) > 0.4, True) & ak.fill_none(events.Jet.delta_r(events.L2_collections) > 0.4, True) & (events.Jet['pt']>30) & ((events.Jet.jetId>>1) & 1) & (abs(events.Jet.eta)<4.7) & (((events.Jet.puId>>2)&1) | (events.Jet['pt']>50))

    for btag_wp in btag_WP:
      events['Jet', f'passDeepJet_{btag_wp}'] = events.Jet.passJet30ID & (abs(events.Jet.eta)<2.5) & (events.Jet.btagDeepFlavB>btag_WP[btag_wp][year])

def JetVeto(events, year, type, vetomap):
    Jets = events.Jet[events.Jet.passJet30ID==1]
    Jets = Jets[(abs(Jets.phi) < 3.1415926536)]
    Electrons = events.Electron
    vetomap = vetomap[f'Summer19UL{year[2:4]}_V1']
    
    if year=='2018':
        jetveto = vetomap.evaluate('jetvetomap_hbp2m1', Jets.eta, Jets.phi) + vetomap.evaluate('jetvetomap_hot', Jets.eta, Jets.phi) # Jets pass if 0, vetoed if nonzero.
        if type=='data': 
            jethem1516 = ak.fill_none(vetomap.evaluate('jetvetomap_hem1516', Jets.eta, Jets.phi), 0) * ak.broadcast_arrays(events.run >= 319077, Jets.pt)[0] # hem1516 only applied for run >= 319077 in data.
            elehem1516 = ak.fill_none(vetomap.evaluate('jetvetomap_hem1516', Electrons.eta, Electrons.phi), 0) * ak.broadcast_arrays(events.run >= 319077, Electrons.eta)[0] 
            jetVetoCut = ak.all((jetveto==0)&(jethem1516==0), axis=1)
            eleVetoCut = ak.all(elehem1516==0, axis=1)
            events = events[jetVetoCut & eleVetoCut]

        elif type=='mc': 
            jethem1516 = ak.fill_none(vetomap.evaluate('jetvetomap_hem1516', Jets.eta, Jets.phi), 0)
            elehem1516 = ak.fill_none(vetomap.evaluate('jetvetomap_hem1516', Electrons.eta, Electrons.phi), 0)
            affected_lumi_factor = (1-((0.016372094+6.891784974+31.835092469)/59.806088326))
            events['genWeight'] = ak.where(ak.any(jethem1516!=0, axis=1)|ak.any(elehem1516!=0, axis=1), events.genWeight*affected_lumi_factor, events.genWeight)
            events = events[ak.all(jetveto==0, axis=1)]

    else:
        jetveto = vetomap.evaluate('jetvetomap', Jets.eta, Jets.phi)
        jetVetoCut = ak.all(jetveto==0, axis=1)
        events = events[jetVetoCut]

    return events

def prettyPrint(arr):
    if isinstance(arr[0], ak.highlevel.Array) or isinstance(arr[0], list):
        if isinstance(ak.flatten(arr)[0], bool):
            return ak.to_list(arr)
        else:
            return [[round(x, 2) for x in (sublist or [])] for sublist in ak.to_list(arr)]
    elif isinstance(arr[0], np.float32):
        return [round(x, 2) if x is not None else None for x in ak.to_list(arr)]
    elif isinstance(arr[0], np.bool):
        return ak.to_list(arr)
    else:
        print(type(arr[0]))