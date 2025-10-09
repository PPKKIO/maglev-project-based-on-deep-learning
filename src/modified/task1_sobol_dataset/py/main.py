import maglev_sim

app = maglev_sim.initialize()
res = app.solve_model4_fem_pkg()
print(res)
app.terminate()
