import awkward as ak
import numpy as np
import copy

def muRochesterCorr(events, type, rochester):
    muons = events.Muon # this is a 2d array
    if type=='data':
        data_k = rochester.kScaleDT(muons.charge, muons.pt, muons.eta, muons.phi) # kScaleDTerror
        muons['pt'] = muons.pt * data_k

    else:
        hasgen = ~np.isnan(ak.fill_none(muons.matched_gen.pt, np.nan))
        mc_kspread = rochester.kSpreadMC(muons.charge, muons.pt, muons.eta, muons.phi, muons.matched_gen.pt) # spread correction if there is gen-level muon (kSpreadMCerror)
        mc_rand = ak.unflatten(np.random.rand(sum(ak.num(muons))), ak.num(muons))
        mc_ksmear = rochester.kSmearMC(muons.charge, muons.pt, muons.eta, muons.phi, muons.nTrackerLayers, mc_rand) # random smear in case of no gen-level muon (kSmearMCerror)
        mc_k = ak.where(hasgen, mc_kspread, mc_ksmear)
        muons['pt'] = muons['pt'] * mc_k
    
    events["Muon"] = muons

def tauCorr(events, type, t_sf):
    if type=='mc':
        dm_veto = (events.Tau.decayMode!=5) & (events.Tau.decayMode!=6) # mode 5 and 6 are experimental -> no SF available
        taus = ak.flatten(events.Tau[dm_veto])
        tau_pt_sf = t_sf["tau_energy_scale"].evaluate(taus.pt, taus.eta, taus.decayMode, taus.genPartFlav, "DeepTau2017v2p1", 'nom')
        mc_k = np.ones(ak.count(events.Tau.pt), dtype=float)
        mc_k[ak.flatten(dm_veto)] = tau_pt_sf
        mc_k = ak.unflatten(mc_k, ak.num(events.Tau))
        events.Tau['pt'] = events.Tau['pt'] * mc_k

def jetCorr(events, year, type, jet_jerc, jer_tool):
    jets = events.Jet
    jets["pt_raw"]    = (1-jets.rawFactor) * jets.pt
    jets["mass_raw"]  = (1-jets.rawFactor) * jets.mass
    jets["event_rho"] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, jets.pt)[0]
    
    # JEC
    jec_map = {
        '2016preVFP': {
            'Run2016B-ver1': 'Summer19UL16APV_RunBCD_V7_DATA_L1L2L3Res_AK4PFchs',
            'Run2016B-ver2': 'Summer19UL16APV_RunBCD_V7_DATA_L1L2L3Res_AK4PFchs',
            'Run2016C-HIPM': 'Summer19UL16APV_RunBCD_V7_DATA_L1L2L3Res_AK4PFchs',
            'Run2016D-HIPM': 'Summer19UL16APV_RunBCD_V7_DATA_L1L2L3Res_AK4PFchs',
            'Run2016E-HIPM': 'Summer19UL16APV_RunEF_V7_DATA_L1L2L3Res_AK4PFchs',
            'Run2016F-HIPM': 'Summer19UL16APV_RunEF_V7_DATA_L1L2L3Res_AK4PFchs',
            'MC': 'Summer19UL16APV_V7_MC_L1L2L3Res_AK4PFchs'
        },
        '2016postVFP': {
            'Run2016F-UL2016': 'Summer19UL16_RunFGH_V7_DATA_L1L2L3Res_AK4PFchs',
            'Run2016G-UL2016': 'Summer19UL16_RunFGH_V7_DATA_L1L2L3Res_AK4PFchs',
            'Run2016H-UL2016': 'Summer19UL16_RunFGH_V7_DATA_L1L2L3Res_AK4PFchs',
            'MC': 'Summer19UL16_V7_MC_L1L2L3Res_AK4PFchs'
        },
        '2017': {
            'Run2017B-UL2017': 'Summer19UL17_RunB_V5_DATA_L1L2L3Res_AK4PFchs',
            'Run2017C-UL2017': 'Summer19UL17_RunC_V5_DATA_L1L2L3Res_AK4PFchs',
            'Run2017D-UL2017': 'Summer19UL17_RunD_V5_DATA_L1L2L3Res_AK4PFchs',
            'Run2017E-UL2017': 'Summer19UL17_RunE_V5_DATA_L1L2L3Res_AK4PFchs',
            'Run2017F-UL2017': 'Summer19UL17_RunF_V5_DATA_L1L2L3Res_AK4PFchs',
            'MC': 'Summer19UL17_V5_MC_L1L2L3Res_AK4PFchs'
        },
        '2018': {
            'Run2018A-UL2018': 'Summer19UL18_RunA_V5_DATA_L1L2L3Res_AK4PFchs',
            'Run2018B-UL2018': 'Summer19UL18_RunB_V5_DATA_L1L2L3Res_AK4PFchs',
            'Run2018C-UL2018': 'Summer19UL18_RunC_V5_DATA_L1L2L3Res_AK4PFchs',
            'Run2018D-UL2018': 'Summer19UL18_RunD_V5_DATA_L1L2L3Res_AK4PFchs',
            'MC': 'Summer19UL18_V5_MC_L1L2L3Res_AK4PFchs'
        }
    }
    jec_key = jec_map[year][events.metadata["dataset"].split('_')[1] if type=='data' else "MC"]
    jec_corrector = jet_jerc.compound[jec_key]
    flat_jet = ak.flatten(jets)
    jec_sf = jec_corrector.evaluate(flat_jet.area, flat_jet.eta, flat_jet.pt_raw, flat_jet.event_rho)
    jec_sf = ak.unflatten(jec_sf, ak.num(jets.area))
    jets['pt'] = jets.pt_raw * jec_sf
    jets['mass'] = jets.mass_raw * jec_sf

    # JER
    if type=='mc':
        jer_map = {
            '2016preVFP': 'Summer20UL16APV_JRV3_MC_',
            '2016postVFP': 'Summer20UL16_JRV3_MC_',
            '2017': 'Summer19UL17_JRV2_MC_',
            '2018': 'Summer19UL18_JRV2_MC_'
        }
        flat_jet = ak.flatten(jets)
        flat_jet["pt_gen"] = ak.fill_none(flat_jet.matched_gen.pt, -1)

        # Evaluating jer_sf
        jer_sf_corrector = jet_jerc[jer_map[year]+'ScaleFactor_AK4PFchs']
        jer_sf = jer_sf_corrector.evaluate(flat_jet.eta, 'nom')

        # Evaluating jer
        jer_corrector = jet_jerc[jer_map[year]+'PtResolution_AK4PFchs']
        jer = jer_corrector.evaluate(flat_jet.eta, flat_jet.pt, flat_jet.event_rho)
        
        # Evaluating jer_smear
        jer_tool = jer_tool['JERSmear']
        run = ak.flatten(ak.broadcast_arrays(events.run, ak.unflatten(jer, ak.num(jets)))[0])
        jer_smear = jer_tool.evaluate(flat_jet.pt, flat_jet.eta, flat_jet.pt_gen , flat_jet.event_rho, run, jer, jer_sf)
        jer_smear = ak.unflatten(jer_smear, ak.num(jets))

        jets['pt'] = jets.pt * jer_smear
        jets['mass'] = jets.mass * jer_smear

    events['Jet'] = jets

def metCorr(events, _type, met_tool):
    met_pt_key  = 'pt_metphicorr_pfmet_'  + _type
    met_phi_key = 'phi_metphicorr_pfmet_' + _type
    met_pt_corrector  = met_tool[met_pt_key]
    met_phi_corrector = met_tool[met_phi_key]

    events['MET', 'pt'] = ak.where(events.MET.pt < 6499, events.MET.pt, 6499) # throws errors from events.MET.pt>6500
    met = events.MET
    npvs = events.PV.npvs

    met['pt']  = met_pt_corrector.evaluate(met.pt, met.phi, npvs, events.run)
    met['phi'] = met_phi_corrector.evaluate(met.pt, met.phi, npvs, events.run)
    events["MET"] = met

def evaluate_bareWeight(events, type):
    if   type=='data': events['weight'] = 1
    elif type=='mc'  : events['weight'] = events.genWeight

def evaluate_btagSF(events, _type, btag_sf, btagWP):
    if _type=='data':
        events["bTagSF"] = ak.where(ak.sum(events.Jet[f'passDeepJet_{btagWP}'],1)==0, 1, 0)
    
    elif _type=='mc':
        if not isinstance(events.Jet.hadronFlavour[0], ak.highlevel.Array):
            events["Jet"] = ak.singletons(events.Jet)
        jet_flat = ak.flatten(events.Jet)
        btagSF_deepjet = np.zeros(len(jet_flat))

        jet_light = ak.where((jet_flat[f'passDeepJet_{btagWP}']) & (jet_flat.hadronFlavour==0))
        jet_heavy = ak.where((jet_flat[f'passDeepJet_{btagWP}']) & (jet_flat.hadronFlavour!=0))

        btagSF_deepjet[jet_light] = btag_sf["deepJet_incl"].evaluate("central", btagWP, jet_flat[jet_light].hadronFlavour, abs(jet_flat[jet_light].eta), jet_flat[jet_light].pt)
        btagSF_deepjet[jet_heavy] = btag_sf["deepJet_comb"].evaluate("central", btagWP, jet_flat[jet_heavy].hadronFlavour, abs(jet_flat[jet_heavy].eta), jet_flat[jet_heavy].pt)
        btagSF_deepjet = ak.unflatten(btagSF_deepjet, ak.num(events.Jet))
        bTagSF = ak.prod(1-btagSF_deepjet, axis=1)
        events['bTagSF'] = bTagSF
    
    events['weight'] = events.weight * events.bTagSF

def evaluate_prefireWeight(events, year, type):
    if (type=='mc') & (year!='2018'):
        events['weight'] = events.weight * events.L1PreFiringWeight.Nom

def evaluate_pileupWeight(events, year, _type, pu_sf):
    if _type=='mc':
        which_year = year[2:4] if '2016' in year else year[2:]
        puWeight = pu_sf[f"Collisions{which_year}_UltraLegacy_goldenJSON"].evaluate(events.Pileup.nTrueInt, "nominal")
        events['weight'] = events.weight * puWeight

def evaluate_ElectronSF(events, year, type, e_sf, e_sf_pri):
    if type=='mc':
        # Electron ID/RECO SF TODO: This is different from Zllf version!!! Check!
        EleReco_SF         = e_sf["UL-Electron-ID-SF"].evaluate(year, "sf", "RecoAbove20", ak.fill_none(events.E_collections.eta, 1), ak.fill_none(events.E_collections.pt, 25)) # Equivalent to previous EleReco_SF * ElelooseIDnoISO_SF
        EleIDnoISO_SF      = e_sf["UL-Electron-ID-SF"].evaluate(year, "sf", "wp80noiso"  , ak.fill_none(events.E_collections.eta, 1), ak.fill_none(events.E_collections.pt, 25))
        #ElelooseIDnoISO_SF = e_sf["UL-Electron-ID-SF"].evaluate(year, "sf", "wplnoiso"  , ak.fill_none(events.E_collections.eta, 1), ak.fill_none(events.E_collections.pt, 25))
        EleISO_SF = e_sf_pri["UL-Electron-ID-SF"].evaluate(year, "sf", "Iso", ak.fill_none(events.E_collections.eta, 1), ak.fill_none(events.E_collections.pt, 25))

        tight_e_events = events.em_ch | events.et_ch | events.eloosem_ch | events.elooset_ch
        loose_e_events = events.looseet_ch | events.looseelooset_ch | events.looseem_ch | events.looseeloosem_ch

        # events['weight'] = events.weight * ak.where(tight_e_events, EleReco_SF * EleIDnoISO_SF     , ak.ones_like(events.weight))
        # events['weight'] = events.weight * ak.where(loose_e_events, EleReco_SF * ElelooseIDnoISO_SF, ak.ones_like(events.weight))
        events['weight'] = events.weight * ak.where(tight_e_events, EleReco_SF * EleIDnoISO_SF     , ak.ones_like(events.weight))
        events['weight'] = events.weight * ak.where(loose_e_events, EleReco_SF, ak.ones_like(events.weight))

        # Electron Trigger SF
        Trig_SF = np.ones(len(events))
        Ele_pass = ak.where((events.etrigger & ~events.mtrigger) & (ak.fill_none(events.E_collections.pt,0) > 29))
        Trig_SF[Ele_pass] = e_sf_pri["UL-Electron-ID-SF"].evaluate(year, "sf", "Trig", events.E_collections[Ele_pass].eta, events.E_collections[Ele_pass].pt) 
        events['weight'] = events.weight * Trig_SF

def evaluate_MuonSF(events, year, _type, m_sf):
    if (_type=='mc') & (sum(ak.num(events.Muon))>0):
        # Muon ID/ISO SF
        MuID_SF      = m_sf["NUM_TightID_DEN_TrackerMuons"].evaluate(abs(ak.fill_none(events.M_collections.eta, 1)), ak.fill_none(events.M_collections.pt, 20), "nominal")
        MulooseID_SF = m_sf["NUM_LooseID_DEN_TrackerMuons"].evaluate(abs(ak.fill_none(events.M_collections.eta, 1)), ak.fill_none(events.M_collections.pt, 20), "nominal")
        MuISO_SF     = m_sf["NUM_TightRelIso_DEN_TightIDandIPCut"].evaluate(abs(ak.fill_none(events.M_collections.eta, 1)), ak.fill_none(events.M_collections.pt, 20), "nominal")

        tight_m_events = events.em_ch | events.mt_ch | events.looseem_ch | events.mlooset_ch
        loose_m_events = events.loosemt_ch | events.eloosem_ch | events.looseeloosem_ch | events.loosemlooset_ch

        events['weight'] = events.weight * ak.where(tight_m_events, MuISO_SF * MuID_SF     , ak.ones_like(events.weight))
        events['weight'] = events.weight * ak.where(loose_m_events, MuISO_SF * MulooseID_SF, ak.ones_like(events.weight))

        # Muon Trigger SF
        if '2016' in year  : triggerstr = 'NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight'
        elif year == '2017': triggerstr = 'NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoTight'
        elif year == '2018': triggerstr = 'NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight'  
        Trig_SF = np.ones(len(events))
        Mu_pass = ak.where(events.mtrigger)
        Trig_SF[Mu_pass] = m_sf[triggerstr].evaluate(abs(events.M_collections[Mu_pass].eta), events.M_collections[Mu_pass].pt, "nominal")
        events['weight'] = events.weight * Trig_SF

def evaluate_TauSF(events, type, t_sf):
    tau_id_sf = np.ones(len(events.T_collections))
    # For some reason, ak.where is suddenly complaining about Nones in an array, demanding them should be either True/False. So I'm filling None -> False. (should I do fill None -> 1?)
    # tau_e      = ak.where(ak.fill_none((events.T_collections.genPartFlav==1) | (events.T_collections.genPartFlav==3), False) & events.et_ch) # returns indices of Trues.
    tau_e      = ak.where(ak.fill_none((events.T_collections.genPartFlav==1) | (events.T_collections.genPartFlav==3), False)) # returns indices of Trues.
    # tau_m      = ak.where(ak.fill_none((events.T_collections.genPartFlav==2) | (events.T_collections.genPartFlav==4), False) & events.mt_ch)
    tau_m      = ak.where(ak.fill_none((events.T_collections.genPartFlav==2) | (events.T_collections.genPartFlav==4), False))
    # tau_h      = ak.where(ak.fill_none(events.T_collections.genPartFlav==5, False) & (events.mt_ch | events.et_ch))
    tau_h      = ak.where(ak.fill_none(events.T_collections.genPartFlav==5, False) & (events.mt_ch | events.et_ch | events.loosemt_ch | events.looseet_ch))
    loosetau_h = ak.where(ak.fill_none(events.T_collections.genPartFlav==5, False) & (events.mlooset_ch | events.elooset_ch | events.loosemlooset_ch | events.looseelooset_ch))

    tau_id_sf[tau_e]      = t_sf["DeepTau2017v2p1VSe"].evaluate(ak.fill_none(events.T_collections[tau_e].eta, 1), ak.fill_none(events.T_collections[tau_e].genPartFlav, 1), 'Tight', 'nom')
    tau_id_sf[tau_m]      = t_sf["DeepTau2017v2p1VSmu"].evaluate(ak.fill_none(events.T_collections[tau_m].eta, 1), ak.fill_none(events.T_collections[tau_m].genPartFlav, 1), 'Tight', 'nom')
    tau_id_sf[tau_h]      = t_sf["DeepTau2017v2p1VSjet"].evaluate(ak.fill_none(events.T_collections[tau_h].pt, 30), ak.fill_none(events.T_collections[tau_h].decayMode, 1), ak.fill_none(events.T_collections[tau_h].genPartFlav, 1), 'Tight', 'Tight', 'nom', 'pt')
    tau_id_sf[loosetau_h] = t_sf["DeepTau2017v2p1VSjet"].evaluate(ak.fill_none(events.T_collections[loosetau_h].pt, 30), ak.fill_none(events.T_collections[loosetau_h].decayMode, 1), ak.fill_none(events.T_collections[loosetau_h].genPartFlav, 1), 'Loose', 'Tight', 'nom', 'pt')
    events['weight'] = events.weight * tau_id_sf