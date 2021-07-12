import matplotlib.pyplot as plt
plt.figure()
plt.plot('./fftloss_list')
plt.savefig('./fft.png',dpi=600)
plt.close()
