"""Open the project's own Plotly views for the full Earth-to-Moon-landing
run, same functions simulation.py has always used (plot_trajectory,
plot_lunar_approach, plot_ascent) -- just pointed at the spliced
full-mission env from full_mission_demo.py instead of a single-phase one.
"""

from full_mission_demo import run_full_demo
from simulation import plot_trajectory, plot_lunar_approach, plot_ascent

env = run_full_demo()

print("\nOpening plots in browser...")

plot_trajectory(env).show()
plot_lunar_approach(env, within_km=20_000.0).show()
plot_ascent(env, t_max=500).show()
