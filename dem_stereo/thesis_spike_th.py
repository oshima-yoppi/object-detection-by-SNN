import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

all_step = 100
spike_rate = 0.3
spk_lst = np.where(np.random.rand(all_step) < spike_rate, 1, 0)
spk_lst = spk_lst * 0.4
print(spk_lst)
# plt.plot(spk_lst)
# plt.show()


class lif:
    def __init__(self, leak, threshold) -> None:
        self.mem = 0
        self.output_lst = []
        self.mem_lst = []
        self.leak = leak
        self.threshold = threshold

    def update(self, input_spike):
        self.mem = self.mem * self.leak + input_spike * 0.5
        self.mem_lst.append(self.mem)
        if self.mem > self.threshold:
            self.mem = 0
            self.output_lst.append(1)
        else:
            self.output_lst.append(0)


IFmodel_low = lif(1.0, 0.5)
IFmodel_mid = lif(1.0, 1)
IFmodel_high = lif(1.0, 1.5)
for i in range(all_step):
    IFmodel_low.update(spk_lst[i])
    IFmodel_mid.update(spk_lst[i])
    IFmodel_high.update(spk_lst[i])


plt.figure(figsize=(4, 6))
plt.subplot(3, 1, 1)
plt.plot(spk_lst)
plt.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
plt.xlabel("time step")
plt.ylabel("Input Spike")
plt.subplot(3, 1, 2)
plt.plot(IFmodel_low.mem_lst)
plt.ylim(0, 1.7)
plt.plot(
    [0, all_step], [IFmodel_low.threshold, IFmodel_low.threshold], linestyle="dashed"
)
plt.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
plt.xlabel("time step")
plt.ylabel("Membrane Potential")
plt.subplot(3, 1, 3)
plt.plot(IFmodel_low.output_lst)
plt.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
plt.xlabel("time step")
plt.ylabel("Output Spike")
plt.tight_layout()
plt.savefig("thesis/spike_IF_th-low.pdf")
plt.show()

plt.close()

plt.figure(figsize=(4, 6))
plt.subplot(3, 1, 1)
plt.plot(spk_lst)
plt.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
plt.xlabel("time step")
plt.ylabel("Input Spike")
plt.subplot(3, 1, 2)
plt.plot(IFmodel_mid.mem_lst)
plt.ylim(0, 1.7)
plt.plot(
    [0, all_step], [IFmodel_mid.threshold, IFmodel_mid.threshold], linestyle="dashed"
)
plt.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
plt.xlabel("time step")
plt.ylabel("Membrane Potential")
plt.subplot(3, 1, 3)
plt.plot(IFmodel_mid.output_lst)
plt.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
plt.xlabel("time step")
plt.ylabel("Output Spike")
plt.tight_layout()
plt.savefig("thesis/spike_IF_th-mid.pdf")
plt.show()
plt.close()

plt.figure(figsize=(4, 6))
plt.subplot(3, 1, 1)
plt.plot(spk_lst)
plt.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
plt.xlabel("time step")

plt.ylabel("Input Spike")
plt.subplot(3, 1, 2)
plt.plot(IFmodel_high.mem_lst)
plt.ylim(0, 1.7)
plt.plot(
    [0, all_step], [IFmodel_high.threshold, IFmodel_high.threshold], linestyle="dashed"
)
plt.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
plt.xlabel("time step")
plt.ylabel("Membrane Potential")
plt.subplot(3, 1, 3)
plt.plot(IFmodel_high.output_lst)

plt.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
plt.xlabel("time step")
plt.ylabel("Output Spike")
plt.tight_layout()
plt.savefig("thesis/spike_IF_th-high.pdf")
plt.show()
