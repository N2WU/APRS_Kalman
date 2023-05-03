import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Load Data

alt_data = np.loadtxt('datasets/w2kgy-12_balloon.csv', delimiter=',', skiprows=1,usecols=6)
time_data = np.loadtxt('datasets/w2kgy-12_balloon.csv', delimiter=',', skiprows=1,usecols=0, dtype=np.datetime64)

alt_data = alt_data*0.3048/1000 #convert to km
data_cutoff = 31
alt_train = alt_data[0:data_cutoff]
alt_est = alt_data[1+data_cutoff:]

t_diff = np.zeros(len(time_data))
for t in range(0,len(time_data)-1):
    t_diff[t] = time_data[t+1]-time_data[t]
T = np.mean(t_diff)

A = np.array([[1, T],[0, 1]])
B = np.array([[0.5*(T**2)],[T]])
N = data_cutoff-1
xn = alt_train-np.min(alt_train)
x_std = np.cov(xn)**2
# KF Parameters
H = np.array([[0,1]])
Rv = np.cov(xn)**2
R_eta = np.eye(1)
# Initialization
y_post = np.array([[0],[1]])
R_post = np.zeros((2,2))

IR = np.eye(np.size(R_post[:,0]))
y_hat = np.zeros(N+1)

for n in range(0,N):
    # Predict
    R_pri = A @ R_post @ A.T + B @ R_eta @ B.T
    y_pri = A @ y_post
    x_pri = H @ y_pri
    # Update
    Rw = H @R_pri @ H.T + Rv
    K = R_pri @ H.T @ np.linalg.inv(Rw)
    y_post = y_pri + K @ (xn[n+1]-x_pri)
    R_post = (IR-K @ H) @ R_pri
    y_hat[n+1] = y_post[1]

#y_hat = [y_hat[1:-1],y_hat[-1]]
# xn = xn[0:-2]
n = np.arange(0,N+1)
xn = np.roll(xn,1)
xn[0] = 0
nx = np.append(0,n)
nx = nx[0:-1]
# plot steps, -> xn and y_hat
fig, ax = plt.subplots()  # Create a figure containing a single axes.
meas_alt = ax.plot(nx[0], xn[0], label='Measured Altitude', color='blue', marker=".",linewidth=2, linestyle='--')[0]  # Plot some data on the axes.
est_alt = ax.plot(n[0],y_hat[0], label='Estimated Altitude', color='orange', marker=".",linestyle=':')[0]
ax.set(xlim=[0, N+1], ylim=[0, np.max(xn)], xlabel='Measurement Step', ylabel='Altitude (km)')
ax.set_title('1D Altitude Kalman Filter')
ax.grid(True)
ax.legend()

def update(frame):
    # update the line plot:
    meas_alt.set_xdata(nx[:frame])
    meas_alt.set_ydata(xn[:frame])
    est_alt.set_xdata(n[:frame])
    est_alt.set_ydata(y_hat[:frame])
    return (meas_alt, est_alt)


ani = animation.FuncAnimation(fig=fig, func=update, frames=N+1, interval=750)
#ani.to_html5_video(64e3)
ani.save('Kalman1D.mp4')
plt.show()
