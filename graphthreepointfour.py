import matplotlib.pyplot as plt

sizes = [64, 128, 256, 512, 1024]
fast_times = [0.00700, 0.02701, 0.16474, 1.32198, 8.68005]
gpu_times = [0.01083, 0.02046, 0.07355, 0.20335, 0.91211]



plt.figure(figsize=(10, 6))
plt.plot(sizes, fast_times, label='Fast Operations', marker='o')
plt.plot(sizes, gpu_times, label='GPU Operations', marker='o')

for i, size in enumerate(sizes):
    plt.text(size, fast_times[i], f'{fast_times[i]:.5f}', fontsize=9, ha='right', va='bottom', color='blue')
    plt.text(size, gpu_times[i], f'{gpu_times[i]:.5f}', fontsize=9, ha='left', va='top', color='orange')

plt.title('Comparison of Fast vs GPU Operations', fontsize=16)
plt.xlabel('Size', fontsize=14)
plt.ylabel('Time (seconds)', fontsize=14)
plt.xscale('log')  
plt.yscale('log')  
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.legend(fontsize=12)
plt.tight_layout()


plt.show()
