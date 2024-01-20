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


IFmodel = lif(1.0, 1)
LIFmodel = lif(0.9, 1)

for i in range(all_step):
    IFmodel.update(spk_lst[i])
    LIFmodel.update(spk_lst[i])

# plt.subplot(3, 2, 1)
# plt.plot(spk_lst)
# plt.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
# plt.subplot(3, 2, 2)
# # メモリなし
# plt.plot(spk_lst)
# plt.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
# plt.subplot(3, 2, 3)
# plt.plot(IFmodel.mem_lst)
# plt.plot([0, all_step], [1, 1], linestyle="dashed")
# plt.ylim(0, 1.2)
# plt.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
# plt.subplot(3, 2, 4)
# plt.plot(LIFmodel.mem_lst)
# plt.plot([0, all_step], [1, 1], linestyle="dashed")
# plt.ylim(0, 1.2)
# plt.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
# plt.subplot(3, 2, 5)
# plt.plot(IFmodel.output_lst)
# plt.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
# plt.subplot(3, 2, 6)
# plt.plot(LIFmodel.output_lst)
# plt.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)

# plt.tight_layout()
# plt.show()

# plt.close()

plt.figure(figsize=(4, 6))
plt.subplot(3, 1, 1)
plt.plot(spk_lst)
plt.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
plt.xlabel("time step")
plt.ylabel("Input Spike")
plt.subplot(3, 1, 2)
plt.plot(IFmodel.mem_lst)
plt.ylim(0, 1.2)
plt.plot([0, all_step], [1, 1], linestyle="dashed")
plt.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
plt.xlabel("time step")
plt.ylabel("Membrane Potential")
plt.subplot(3, 1, 3)
plt.plot(IFmodel.output_lst)
plt.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
plt.xlabel("time step")
plt.ylabel("Output Spike")
plt.tight_layout()
plt.savefig("thesis/spike_IF.pdf")
plt.show()

plt.close()

plt.figure(figsize=(4, 6))
plt.subplot(3, 1, 1)
plt.plot(spk_lst)
plt.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
plt.xlabel("time step")
plt.ylabel("Input Spike")
plt.subplot(3, 1, 2)
plt.plot(LIFmodel.mem_lst)
plt.ylim(0, 1.2)
plt.plot([0, all_step], [1, 1], linestyle="dashed")
plt.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
plt.xlabel("time step")
plt.ylabel("Membrane Potential")
plt.subplot(3, 1, 3)
plt.plot(LIFmodel.output_lst)
plt.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
plt.xlabel("time step")
plt.ylabel("Output Spike")
plt.tight_layout()
plt.savefig("thesis/spike_LIF.pdf")
plt.show()
