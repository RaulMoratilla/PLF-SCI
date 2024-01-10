out = sim("ackerman_ROS_fuzzy_controller.slx");
data = [out.sens, out.WV];
writematrix(data, "datos_neuronal/ej15");