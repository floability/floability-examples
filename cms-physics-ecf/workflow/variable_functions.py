import fastjet
import awkward as ak
import coffea.nanoevents.methods.vector as vector
import numpy as np


def num_subjets(fatjet, cluster_val):
    pf = ak.flatten(fatjet.constituents.pf, axis=1)
    jetdef = fastjet.JetDefinition(fastjet.cambridge_algorithm, cluster_val)
    cluster = fastjet.ClusterSequence(pf, jetdef)
    subjets = cluster.inclusive_jets()
    num = ak.num(subjets, axis=1)
    return num

def color_ring(fatjet, cluster_val):
    pf = ak.flatten(fatjet.constituents.pf, axis=1)
    jetdef = fastjet.JetDefinition(fastjet.cambridge_algorithm, cluster_val)
    cluster = fastjet.ClusterSequence(pf, jetdef)
    #subjets = cluster.exclusive_jets(n_jets=3)
    subjets = cluster.inclusive_jets()
    vec = ak.zip(
        {
            "x": subjets.px,
            "y": subjets.py,
            "z": subjets.pz,
            "t": subjets.E,
        },
        with_name="LorentzVector",
        behavior=vector.behavior,
    )
    vec = ak.pad_none(vec, 3)
    vec["norm3"] = np.sqrt(vec.dot(vec))
    vec["idx"] = ak.local_index(vec)
    i, j = ak.unzip(ak.combinations(vec, 2))
    best = ak.argmax(abs((i + j).pt), axis=1, keepdims=True)
    order_check = ak.concatenate([i[best].pt, j[best].pt], axis=1)
    leading = ak.argmax(order_check, axis=1, keepdims=True)
    subleading = ak.argmin(order_check, axis=1, keepdims=True)
    leading_particles = ak.concatenate([i[best], j[best]], axis=1)
    cut = ((vec.idx != ak.firsts(leading_particles.idx)) & 
       (vec.idx != ak.firsts(ak.sort(leading_particles.idx, ascending=False)))
      )
    everything_else = vec[cut]
    total_pt = ak.sum(vec.pt, axis=1)
    everything_else['momentum_fraction'] = (everything_else.pt)/total_pt
    everything_else['weighted_eta'] = everything_else.eta * everything_else.momentum_fraction
    everything_else['weighted_phi'] = everything_else.phi * everything_else.momentum_fraction
    weighted_average_eta = ak.sum(everything_else.weighted_eta, axis=1)/ak.num(everything_else, axis=1)
    weighted_average_phi = ak.sum(everything_else.weighted_phi, axis=1)/ak.num(everything_else, axis=1)
    leg1 = leading_particles[leading]
    leg2 = leading_particles[subleading]
    leg1 = ak.firsts(leg1)
    leg2 = ak.firsts(leg2)
    a13 = ((((leg1.eta * (leg1.pt/total_pt)
             ) - weighted_average_eta)**2) + (((leg1.phi * (leg1.pt/total_pt)
                                      ) - weighted_average_phi)**2))
    a23 = ((((leg2.eta * (leg2.pt/total_pt)
             ) - weighted_average_eta)**2) + (((leg2.phi * (leg2.pt/total_pt)
                                      ) - weighted_average_phi)**2))
    a12 = ((((leg1.eta * (leg1.pt/total_pt)
             ) - (leg2.eta * (leg2.pt/total_pt)
                 ))**2) + (((leg1.phi * (leg1.pt/total_pt)
                            ) - (leg2.phi * (leg2.pt/total_pt)
                                ))**2))
    color_ring = (a13 + a23) / (a12)

    return color_ring

def d2_calc(fatjet):
    jetdef = fastjet.JetDefinition(
        fastjet.cambridge_algorithm, 0.8
    )
    pf = ak.flatten(fatjet.constituents.pf, axis=1)
    cluster = fastjet.ClusterSequence(pf, jetdef)
    softdrop = cluster.exclusive_jets_softdrop_grooming()
    softdrop_cluster = fastjet.ClusterSequence(softdrop.constituents, jetdef)
    d2 = softdrop_cluster.exclusive_jets_energy_correlator(func="D2")
    return d2

def n4_calc(fatjet):
    jetdef = fastjet.JetDefinition(
        fastjet.cambridge_algorithm, 0.8
    )
    pf = ak.flatten(fatjet.constituents.pf, axis=1)
    cluster = fastjet.ClusterSequence(pf, jetdef)
    softdrop = cluster.exclusive_jets_softdrop_grooming()
    softdrop_cluster = fastjet.ClusterSequence(softdrop.constituents, jetdef)
    n4 = softdrop_cluster.exclusive_jets_energy_correlator(
        func="nseries", npoint=4
    )
    return n4

def m2_calc(fatjet):
    jetdef = fastjet.JetDefinition(
        fastjet.cambridge_algorithm, 0.8
    )
    pf = ak.flatten(fatjet.constituents.pf, axis=1)
    cluster = fastjet.ClusterSequence(pf, jetdef)
    softdrop = cluster.exclusive_jets_softdrop_grooming()
    softdrop_cluster = fastjet.ClusterSequence(softdrop.constituents, jetdef)
    m2 = softdrop_cluster.exclusive_jets_energy_correlator(
        func="mseries", npoint=2
    )
    return m2

def m3_calc(fatjet):
    jetdef = fastjet.JetDefinition(
        fastjet.cambridge_algorithm, 0.8
    )
    pf = ak.flatten(fatjet.constituents.pf, axis=1)
    cluster = fastjet.ClusterSequence(pf, jetdef)
    softdrop = cluster.exclusive_jets_softdrop_grooming()
    softdrop_cluster = fastjet.ClusterSequence(softdrop.constituents, jetdef)
    m3 = softdrop_cluster.exclusive_jets_energy_correlator(
        func="mseries", npoint=3
    )
    return m3

def d3_calc(fatjet):
    jetdef = fastjet.JetDefinition(
        fastjet.cambridge_algorithm, 0.8
    )
    pf = ak.flatten(fatjet.constituents.pf, axis=1)
    cluster = fastjet.ClusterSequence(pf, jetdef)
    softdrop = cluster.exclusive_jets_softdrop_grooming()
    softdrop_cluster = fastjet.ClusterSequence(softdrop.constituents, jetdef)
    e4 = softdrop_cluster.exclusive_jets_energy_correlator(
        func="generic", normalized=True, npoint=4,
    )
    e2 = softdrop_cluster.exclusive_jets_energy_correlator(
        func="generic", normalized=True, npoint=2,
    )
    e3 = softdrop_cluster.exclusive_jets_energy_correlator(
        func="generic", normalized=True, npoint=3,
    )
    d3 = (e4 * (e2**3)) / (e3**3)
    return d3

def u_calc(fatjet, n):
    jetdef = fastjet.JetDefinition(
        fastjet.cambridge_algorithm, 0.8
    )
    pf = ak.flatten(fatjet.constituents.pf, axis=1)
    cluster = fastjet.ClusterSequence(pf, jetdef)
    softdrop = cluster.exclusive_jets_softdrop_grooming()
    softdrop_cluster = fastjet.ClusterSequence(softdrop.constituents, jetdef)
    if n == 1:
        u = softdrop_cluster.exclusive_jets_energy_correlator(func="u1")
    if n == 2:
        u = softdrop_cluster.exclusive_jets_energy_correlator(func="u2")
    if n == 3:
        u = softdrop_cluster.exclusive_jets_energy_correlator(func="u3")
    return u

def make_ecf(fatjet, n, v, b):
    jetdef = fastjet.JetDefinition(
        fastjet.cambridge_algorithm, 0.8
    )
    pf = ak.flatten(fatjet.constituents.pf, axis=1)
    cluster = fastjet.ClusterSequence(pf, jetdef)
    softdrop = cluster.exclusive_jets_softdrop_grooming()
    softdrop_cluster = fastjet.ClusterSequence(softdrop.constituents, jetdef)
    ecf = softdrop_cluster.exclusive_jets_energy_correlator(
        func='generic', npoint=n, angles=v, beta=b, normalized=True,
    )
    return ecf