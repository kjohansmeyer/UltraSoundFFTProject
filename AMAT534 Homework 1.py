# --------------------------------------------------------------------------- #
# Name: Kevin Johansmeyer                                                     #
# Course: AMAT-534: Data-Driven Modeling and Computation                      #
# Professor: Dr. Eric Forgoston                                               #
# Assignment: Homework #1                                                     #
# --------------------------------------------------------------------------- #

# ------------------------ Provided Code (modified)-------------------------- # 
import numpy as np
from scipy.io import loadmat
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = "browser" # Opens Plotly plot in web browser

usd = loadmat('ultrasounddata.mat')
Undata = usd['Undata']

# Setting up spacial dimensions and arrays
zeroCube = np.zeros((64,64,64), complex) #64x64x64 cube with zeros
Un = zeroCube
L = 15 # spatial domain
n = 64 # Fourier modes
x2 = np.linspace(-L, L, n+1)
x = x2[0:n]
y = x
z = x
[X,Y,Z]=np.meshgrid(x,y,z) # cube grid on [-15,15] for each dimension

r=int(n/2-1)
s=int(n/2)
k1=(2*np.pi/(2*L))*np.linspace(0, r, s)
k2=(2*np.pi/(2*L))*np.linspace(-s, -1, s)
k=np.append(k1,k2)
ks=np.fft.fftshift(k)
[Kx,Ky,Kz]=np.meshgrid(ks,ks,ks) # cube grid in frequency space

# Undata.shape = (20,262144) = (20,64x64x64) [20 rows of 64x64x64 complex values]
# Data reshaped into Un.shape = (64,64,64) [20 64x64x64 cubes]

# Code modified to reference each individual scan
scans = ['scan0','scan1','scan2','scan3','scan4','scan5','scan6','scan7',
         'scan8','scan9','scan10','scan11','scan12','scan13','scan14',
         'scan15','scan16','scan17','scan18','scan19']

sumUn = zeroCube
for i in range(0,20):
    scans[i] = np.reshape(Undata[i,:],(n,n,n))
    sumUn = np.add(scans[i],sumUn) #sum all 20 scans
    
# ------------------- Plotting the Average of the 20 Scans ------------------ #
avgUn = sumUn/20
modAvgUn = np.abs(avgUn)

# Plotting the Modulus of the Average of 20 Cubes in Physical Space:
# Citation: https://plotly.com/python/3d-isosurface-plots/
fig = go.Figure(data=go.Isosurface(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=modAvgUn.flatten(),
    isomin=0.0,
    isomax=0.1,
    caps=dict(x_show=True, y_show=True)
    ))
# fig.show() #uncomment to show 3D Isosurface (in browser)

# ---------------- Fast-Fourier Transforming Each Scan ---------------------- #

FFTs = ['FFT0','FFT1','FFT2','FFT3','FFT4','FFT5','FFT6','FFT7',
          'FFT8','FFT9','FFT10','FFT11','FFT12','FFT13','FFT14',
          'FFT15','FFT16','FFT17','FFT18','FFT19']

sumFFTUn = zeroCube
for i in range(0,20):
    FFTs[i] = np.fft.fftn(scans[i]) #FFT each of the 20 scans
    sumFFTUn = np.add(FFTs[i],sumFFTUn) #sum all 20 scans

# Averaging all 20 scans
avgFFTUn = sumFFTUn/20
modAvgFFTUn = np.abs(avgFFTUn)

# ------------------------- Finding Maximum Intensity ----------------------- #

# Function for finding maximum value and corresponding indices of 3D array
def max_val_and_index(array3D):
    max_val = 0
    for i in range(0,n):
        for j in range(0,n):
            for k in range(0,n):
                if array3D[i][j][k] > max_val:
                    max_val = array3D[i][j][k]
                    # indices switched to match Gaussian Function's notation:
                    max_val_index = j,i,k 
    print("Maximum value:",max_val)
    print("Indices for maximum value:",max_val_index)
    return max_val_index

print("modAvgFFTUn:")
max_val_and_index(modAvgFFTUn)

print("Center Frequency Components:")
print("x:",ks[0])
print("y:",ks[9])
print("z:",ks[59])
print("")

# --------------- Filtering Around the Center Frequency --------------------- #

# The peak should be at (a,b,c), NOT (b,a,c)
a = ks[0]
b = ks[9] # Be careful with indices and measurements in frequency space
c = ks[59]

sigmaX = 0.1
sigmaY = 0.1
sigmaZ = 0.1

# --------------- Applying the Filter Around Center Frequency --------------- #
gaussianFilter = np.exp(- sigmaX*(Kx-a)**2 - sigmaY*(Ky-b)**2 - sigmaZ*(Kz-c)**2)


filteredFFTs = ['filteredFFT0','filteredFFT1','filteredFFT2','filteredFFT3',
                'filteredFFT4','filteredFFT5','filteredFFT6','filteredFFT7',
                'filteredFFT8','filteredFFT9','filteredFFT10','filteredFFT11',
                'filteredFFT12','filteredFFT13','filteredFFT14',
                'filteredFFT15','filteredFFT16','filteredFFT17',
                'filteredFFT18','filteredFFT19']

for i in range(0,20):
    filteredFFTs[i] = FFTs[i] * gaussianFilter
    
modIFFTs = ['IFFT0','IFFT1','IFFT2','IFFT3','IFFT4','IFFT5','IFFT6','IFFT7',
          'IFFT8','IFFT9','IFFT10','IFFT11','IFFT12','IFFT13','IFFT14',
          'IFFT15','IFFT16','IFFT17','IFFT18','IFFT19']

marbleX = np.zeros(20)
marbleY = np.zeros(20)
marbleZ = np.zeros(20)

for i in range(0,20):
    modIFFTs[i] = np.abs(np.fft.ifftn(filteredFFTs[i]))
    
    print("Scan #",i,":")
    x_index,y_index,z_index = max_val_and_index(modIFFTs[i])
    print("Position:",x[x_index],",",y[y_index],",",z[z_index])
    print("")

    marbleX[i] = np.array(x[x_index])
    marbleY[i] = np.array(y[y_index])
    marbleZ[i] = np.array(z[z_index])

# Citation: https://plotly.com/python/3d-isosurface-plots/
fig = go.Figure(data=go.Isosurface(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=modIFFTs[0].flatten(),
    isomin=0.0,
    isomax=0.2,
    caps=dict(x_show=False, y_show=True)
    ))

fig.add_trace(
    go.Scatter3d(x=marbleX, 
                 y=marbleY, 
                 z=marbleZ,
                 mode='markers'))

# fig.show() #uncomment to show 3D Isosurface (in browser)

# Citation: https://plotly.com/python/3d-line-plots/
fig = go.Figure(data=go.Scatter3d(
    x=marbleX, 
    y=marbleY, 
    z=marbleZ,
    marker=dict(
        size=4,
        color=z,
        colorscale='Viridis',
    ),
    line=dict(
        color='darkblue',
        width=2
    )
))
# Citation: https://plotly.com/python/3d-axes/
fig.update_layout(
    scene = dict(
        xaxis = dict(nticks=30, range=[-15,15],),
                     yaxis = dict(nticks=30, range=[-15,15],),
                     zaxis = dict(nticks=30, range=[-15,15],),),
    width=700,
    scene_aspectmode='cube',
    margin=dict(r=20, l=10, b=10, t=10))
fig.show()