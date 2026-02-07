import sys
import numpy as np
import awkward as ak
from coffea.nanoevents.methods import candidate

def Rpt(lep1, lep2, jets=None):
    dilep = lep1+lep2
    if jets==None:
        return (dilep).pt/(lep1.pt+lep2.pt)
    elif len(jets)==1:
        return (dilep + jets[0]).pt/(dilepVar.pt+jets[0].pt)
    elif len(jets)==2:
        return (dilep + jets[0] +jets[1]).pt/(dilepVar.pt+jets[0].pt+jets[1].pt)
    else:
        return -999
    
def pZeta(leg1, leg2, MET_px, MET_py):
    leg1x = np.cos(leg1.phi)
    leg2x = np.cos(leg2.phi)
    leg1y = np.sin(leg1.phi)
    leg2y = np.sin(leg2.phi)
    zetaX = leg1x + leg2x
    zetaY = leg1y + leg2y
    zetaR = np.sqrt(zetaX*zetaX + zetaY*zetaY)
    
    zetaX = np.where((zetaR > 0.), zetaX/zetaR, zetaX)
    zetaY = np.where((zetaR > 0.), zetaY/zetaR, zetaY)
    
    visPx = leg1.px + leg2.px
    visPy = leg1.py + leg2.py
    pZetaVis = visPx*zetaX + visPy*zetaY
    px = visPx + MET_px
    py = visPy + MET_py
    
    pZeta = px*zetaX + py*zetaY
    
    return (pZeta, pZetaVis)

def Zeppenfeld(lep1, lep2, jets):
    dilep = lep1+lep2
    if len(jets)==1:
        return dilep.eta - (jets[0].eta)/2
    elif len(jets)==2:
        return dilep.eta - (jets[0].eta + jets[1].eta)/2
    else:
        return -999
    
# def mT(met, lep):
#     d = met.phi - lep.phi
#     dphi = (d + np.pi) % (2*np.pi) - np.pi
#     return np.sqrt(2.0 * lep.pt * met.pt * (1.0 - np.cos(dphi)))

def mT(pt1, phi1, pt2, phi2):
    dphi = (phi1 - phi2 + np.pi) % (2*np.pi) - np.pi
    return np.sqrt(2.0 * pt1 * pt2 * (1.0 - np.cos(dphi)))

def mT3(met, lep1, lep2):
    lep12 = lep1+lep2
    return np.sqrt(abs((np.sqrt(lep12.mass**2+lep12.pt**2) + met.pt)**2 - (lep12+met).pt**2))

def pt_cen(lep1, lep2, jets):
    dilep = lep1+lep2
    if len(jets)==1:
        return dilep.pt - jets[0].pt/2
    elif len(jets)==2:
        return dilep.pt - (jets[0] + jets[1]).pt/2
    else:
        return -999

def kinematics(events, tp, doSys=False):
    Jets = events.Jet[events.Jet.passJet30ID>0]
    Jets = Jets[ak.argsort(Jets.pt, axis=1, ascending=False)] # sorted by leading pT
    events["nJets"] = ak.num(Jets)
    # GenJet information
    if tp=='mc':
        events['GenJet_pt']   = events.GenJet.pt
        events['GenJet_eta']  = events.GenJet.eta
        events['GenJet_phi']  = events.GenJet.phi
        events['GenJet_mass'] = events.GenJet.mass
        events['GenJetIdx']   = Jets.genJetIdx
    elif tp=='data':
        events['GenJet_pt']   = ak.full_like(Jets.pt, -999)
        events['GenJet_eta']  = ak.full_like(Jets.pt, -999)
        events['GenJet_phi']  = ak.full_like(Jets.pt, -999)
        events['GenJet_mass'] = ak.full_like(Jets.pt, -999)
        events['GenJetIdx']   = ak.full_like(Jets.pt, -999)

    events["Jet1"] = ak.firsts(Jets)       # leading jet in pT
    events["Jet2"] = ak.firsts(Jets[:,1:]) # subleading jet in pT
    # need to correctly select VBF jets, not simply leading and subleading in pT. Leading 5 jets in pT.
    # for signal mc, we need to save the gen vars.
    # GenJet_partonFlavour -> just save the number

    # Jet information
    events["J1_lab_pt"] = events.Jet1.pt   ; events["J2_lab_pt"] = events.Jet2.pt
    events["J1_eta"]    = events.Jet1.eta  ; events["J2_eta"]    = events.Jet2.eta
    events["J1_phi"]    = events.Jet1.phi  ; events["J2_phi"]    = events.Jet2.phi
    events["J1_mass"]   = events.Jet1.mass ; events["J2_mass"]   = events.Jet2.mass

    # Dijet information
    events["J1_J2_DeltaEta"] = events.Jet1.eta - events.Jet2.eta
    events["Mjj"] = (events.Jet1 + events.Jet2).mass
    events["passVBFcut"] = ak.fill_none((events.Mjj>550), False)

    if 'Tau_DM' not in events.fields:
        events['Tau_DM'] = ak.fill_none(events.T_collections.decayMode, -999)

    #compute interesting variables
    dilep = events['L1_collections'] + events['L2_collections']

    #For e_mu channel, decide which lepton comes from the tau 
    #by checking highest momentum in Higgs frame
    MET_with_dilep_eta = ak.zip(
        {
            "pt": events.MET.pt,
            "eta": dilep.eta,
            "phi": events.MET.phi,
            "mass": ak.zeros_like(events.MET.pt),
            "charge": ak.zeros_like(events.MET.pt),
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )

    MET = events.MET

    H = dilep + MET_with_dilep_eta #H is already a PtEtaPhiMCandidate

    events["dilep_mass"] = dilep.mass
    events["dilep_pt"] = dilep.pt

    events["L1_lab_pt"] = events['L1_collections'].pt
    events["L2_lab_pt"] = events['L2_collections'].pt
    events["L1_eta"] = events['L1_collections'].eta
    events["L2_eta"] = events['L2_collections'].eta
    events["L1_M"] = events['L1_collections'].mass
    events["L2_M"] = events['L2_collections'].mass
    events["L1_genPartFlav"] = events['L1_collections'].genPartFlav
    events["L2_genPartFlav"] = events['L2_collections'].genPartFlav

    events["L1_L2_DeltaEta"] = abs(events['L1_collections'].eta - events['L2_collections'].eta)
    events["L1_L2_DeltaPhi"] = abs((events['L1_collections'].phi - events['L2_collections'].phi + np.pi) % (2 * np.pi) - np.pi)

    # print("events['L1_collections'].phi", events['L1_collections'].phi)
    # print("events['L2_collections'].phi", events['L2_collections'].phi)
    # print('events["L1_L2_DeltaPhi"]', events["L1_L2_DeltaPhi"])

    events["L1_H_pt"] = (events['L1_collections'].boost(-H.boostvec)).pt
    events["L1_H_p"] = (events['L1_collections'].boost(-H.boostvec)).pvec.rho
    events["L2_H_pt"] = (events['L2_collections'].boost(-H.boostvec)).pt
    events["L2_H_p"] = (events['L2_collections'].boost(-H.boostvec)).pvec.rho

    events["col_mass_L1"] = events["dilep_mass"]/np.sqrt(events['L1_collections'].pt/(MET.pt+events['L1_collections'].pt))
    events["col_mass_L2"] = events["dilep_mass"]/np.sqrt(events['L2_collections'].pt/(MET.pt+events['L2_collections'].pt)) #assuming L2 is tau decay products

    #For tau_h channels, L1 is always e/mu and L2 the tau
    #For e_mu channel, L1 is the lepton leading in the boosted frame



    events["met"] = MET.pt
    #events["L1_met_mT"] = mT(MET, events['L1_collections'])
    events["L1_met_mT"] = mT(MET.pt, MET.phi, events.L1_collections.pt, events.L1_collections.phi)
    # events["L2_met_mT"] = mT(MET, events['L2_collections'])
    events["L2_met_mT"] = mT(MET.pt, MET.phi, events.L2_collections.pt, events.L2_collections.phi)
    events["L1_met_DeltaPhi"] = events.L1_collections.delta_phi(MET)
    events["L2_met_DeltaPhi"] = events.L2_collections.delta_phi(MET)