"""
This module is part of the SIENA-IH-STORM model

Functions described here are part of the data pre-processing.
Strongly inspired from 2024 Itxaso Odériz.

Copyright (C) 2026 Mathys Baldacchino.


"""

import os
import sys
import CODE.import_data as import_data
import CODE.climatology as climatology


__location__ = os.path.realpath(os.getcwd())  # TEMP FIX?
dir_path = __location__


(
    period,
    climate_index,
    threshold,
    idx_basin,
    months,
    mpi_bounds,
    months_for_coef_MPI,
    months_for_coef_PRESS,
    generation_phase,
    genesis_weighting,
) = import_data.input_data("input.dat")

print("period:", period)
print("climate_index:", climate_index)
print("threshold:", threshold)
print("idx_basin:", idx_basin)
print("months:", months)
print("mpi_bounds:", mpi_bounds)
print("genesis method:", genesis_weighting)

# Always prepare the all-years inputs first.
climatology.climatology_data(period)

# Then derive pooled + phase-aware climatologies from the selected index.
if climate_index.lower() != "none":
    climate_df = climatology.build_pooled_and_phase_climatologies(
        period, climate_index=climate_index, threshold=threshold
    )

    # Save individual year-month fields for interannual resampling
    climatology.save_yearly_env_fields(climate_df, period)

# Genesis GLM auxiliary climate predictors (separate from env_yearly/).
# Must follow save_yearly_env_fields, as JASO reads SST_<year>_<month>.npy.
if genesis_weighting == "GLM":
    from CODE.genesis_complementary import build_historical_features

    selected = ["EP", "NA", "NI", "SI", "SP", "WP"]
    needed = [selected[i] for i in idx_basin if selected[i] in ("NA", "EP", "WP", "SP")]
    if needed:
        build_historical_features(__location__, period=period, basins=needed)
        print("Historical GLM complementary features saved")
