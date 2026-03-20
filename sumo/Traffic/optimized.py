import os
import sys
import traci

# SUMO setup
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("SUMO_HOME not set")

sumoCmd = ["sumo-gui", "-c", "simulation.sumocfg", "--start"]
traci.start(sumoCmd)

tlsID = traci.trafficlight.getIDList()[0]

# Get directional queues
def get_directional_queue():
    lanes = traci.trafficlight.getControlledLanes(tlsID)

    ns, ew = 0, 0

    for lane in lanes:
        count = traci.lane.getLastStepVehicleNumber(lane)

        if "N" in lane or "S" in lane:
            ns += count
        else:
            ew += count

    return ns, ew

step = 0

while step < 1000:
    traci.simulationStep()

    ns, ew = get_directional_queue()

    # AI DECISION
    if step % 10 == 0:
        if ns > ew:
            traci.trafficlight.setPhase(tlsID, 0)
        else:
            traci.trafficlight.setPhase(tlsID, 1)

    print(f"[AI] Step {step} | NS: {ns} | EW: {ew}")

    step += 1

traci.close()
