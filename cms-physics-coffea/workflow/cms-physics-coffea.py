from coffea.nanoevents import NanoEventsFactory
from coffea import processor

import dask_awkward as dak
import hist.dask as hda


use_taskvine = True

if use_taskvine:
    from ndcctools.taskvine import DaskVine

    vine_manager = DaskVine(port=9500, name="saiful-agc")

    executor_args = {
        "scheduler": vine_manager,
        "worker_transfers": True,
        "task_mode": "function-calls",
    }
else:
    from distributed import Client

    client = Client()

    executor_args = {}


data_file = ("file:///scratch365/btovar/data//Run2012B_SingleMu.root",)

events = NanoEventsFactory.from_root(
    {data_file: "/Events"}, metadata={"dataset": "SingleMu"}
).events()

q1_hist = (
    hda.Hist.new.Reg(100, 0, 200, name="met", label="$E_{T}^{miss}$ [GeV]")
    .Double()
    .fill(events.MET.pt)
)

q1_hist.compute(**executor_args).plot1d()

dak.necessary_columns(q1_hist)
