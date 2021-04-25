# Imports

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D 

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from scipy.stats import gaussian_kde
import scipy.stats as st
import os

# Utility Functions

# Utility functions

def print_input_files():
    # Input data files are available in the "../input/" directory.
    for dirname, _, filenames in os.walk('/content/'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

# Dump text files
def dump_text_file(fname):
    with open(fname, 'r') as f:
        print(f.read())

# Dump CSV files
def dump_csv_file(fname, count=5):
    # count: 0 - column names only, -1 - all rows, default = 5 rows max
    df = pd.read_csv(fname)
    if count < 0:
        count = df.shape[0]
    return df.head(count)

# Dataset related functions
ds_nbaiot = '/content/'
dn_nbaiot = ['Danmini_Doorbell', 'Ecobee_Thermostat', 'Ennio_Doorbell', 'Philips_B120N10_Baby_Monitor', 'Provision_PT_737E_Security_Camera', 'Provision_PT_838_Security_Camera', 'Samsung_SNH_1011_N_Webcam', 'SimpleHome_XCS7_1002_WHT_Security_Camera', 'SimpleHome_XCS7_1003_WHT_Security_Camera']

def fname(ds, f):
    if '.csv' not in f:
        f = f'{f}.csv'
    return os.path.join(ds, f)

def fname_nbaiot(f):
    return fname(ds_nbaiot, f)

def get_nbaiot_device_files():
    nbaiot_all_files = dump_csv_file(fname_nbaiot('data_summary'), -1)
    nbaiot_all_files = nbaiot_all_files.iloc[:,0:1].values
    device_id = 1
    indices = []
    for j in range(len(nbaiot_all_files)):
        if str(device_id) not in str(nbaiot_all_files[j]):
            indices.append(j)
            device_id += 1
    nbaiot_device_files = np.split(nbaiot_all_files, indices)
    return nbaiot_device_files

def get_nbaiot_device_data(device_id, count_norm=-1, count_anom=-1):
    if device_id < 1 or device_id > 9:
        assert False, "Please provide a valid device ID 1-9, both inclusive"
    if count_anom == -1:
        count_anom = count_norm
    device_index = device_id -1
    device_files = get_nbaiot_device_files()
    device_file = device_files[device_index]
    df = pd.DataFrame()
    y = []
    for i in range(len(device_file)):
        fname = str(device_file[i][0])
        df_c = pd.read_csv(fname_nbaiot(fname))
        count = count_anom
        if 'benign' in fname:
            count = count_norm
        rows = count if count >=0 else df_c.shape[0]
        print("processing", fname, "rows =", rows)
        y_np = np.ones(rows) if 'benign' in fname else np.zeros(rows)
        y.extend(y_np.tolist())
        df = pd.concat([df.iloc[:,:].reset_index(drop=True),
                      df_c.iloc[:rows,:].reset_index(drop=True)], axis=0)
    X = df.iloc[:,:].values
    y = np.array(y)
    return (X, y)

def get_nbaiot_devices_data():
    devices_data = []
    for i in range(9):
        device_id = i + 1
        (X, y) = get_nbaiot_device_data(device_id)
        devices_data.append((X, y))
    return devices_data


# Bilinear maps
def normalize(X, x_min=0, x_max=1):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom

class BilinearMap:
    def __init__(self, target_n):
        self.target_cols = target_n

    def compute_coeff(self, X, y):
        try:
            Xt = np.transpose(X)
            Xp = np.dot(Xt, X)
            Xpi = np.linalg.inv(Xp)
            XpiXt = np.dot(Xpi, Xt)
            coeff = np.dot(XpiXt, y)
            print ('coeff.shape:', coeff.shape)
        except Exception as e:
            regressor = LinearRegression(fit_intercept=False)
            regressor.fit(X, y)
            coeff = regressor.coef_
            print ('Exception:', e)

        return coeff

    def fit_transform(self, X, y):
        target_rows = X.shape[1]
        actual_rows = X.shape[0]
        required_rows = target_rows * self.target_cols

        if actual_rows < required_rows:
            assert False, f"{required_rows} rows are required, {actual_rows} are provided"

        Y = []
        for i in range(self.target_cols):
            start = i * target_rows
            end = start + target_rows
            coeff = self.compute_coeff(X[start:end,:], y[start:end])
            Y.extend(coeff.tolist())
            print("coeff.shape:", coeff.shape, "Len y:", len(Y), 'Start:', start, 'End:', end)
        Y = np.array(Y)
        Y = Y.reshape(target_rows, self.target_cols)
        print("Y.shape:", Y.shape)
        Z = np.dot(X, Y)
        return Z


 # Visualisation Utils

def plot_scatter_nbaiot_device(device_data, device_id, dim3=True):
    if device_id < 1 or device_id > 9:
        assert False, "Please provide a valid device ID 1-9, both inclusive"
    device_index = device_id-1
    print("scatter plot for", dn_nbaiot[device_index])
    (X, y) = device_data
    X_std = StandardScaler().fit_transform(X)

    bmap = PCA(n_components=2)
    X_bmap = bmap.fit_transform(X_std)

    print("X_bmap.shape:", X_bmap.shape, "X_std.shape:", X_std.shape)
    data_X = X_bmap[:,0]
    data_Y = X_bmap[:,1]
    data_Z = y
    data = np.column_stack((data_X, data_Y, data_Z))
    #if dim3:
    plot_3d_scatter(data, dn_nbaiot[device_index], 'PCA1', 'PCA2', 'Normal or Anomalous')
    #else:
    normal = mpatches.Patch(color='green', label='N')
    anomalous = mpatches.Patch(color='red', label='A')
    handles = [normal, anomalous]
    plot_2d_scatter(data, dn_nbaiot[device_index], 'PCA1', 'PCA2', handles)

def plot_surface_nbaiot_device(device_data, device_id):
    if device_id < 1 or device_id > 9:
        assert False, "Please provide a valid device ID 1-9, both inclusive"
    device_index = device_id-1
    print("scatter plot for", dn_nbaiot[device_index])
    (X, y) = device_data
    X_std = StandardScaler().fit_transform(X)

    bmap = PCA(n_components=2)
    X_bmap = bmap.fit_transform(X_std)

    print("X_bmap.shape:", X_bmap.shape, "X_std.shape:", X_std.shape)
    plot_3d_scatter_surface(X_bmap, dn_nbaiot[device_index], 'PCA1', 'PCA2', 'PCA3')

# Visualization related functions
def plot_3d_histogram(data):
    cols = data.shape[1]
    if cols < 2:
        assert False, 'The number of columns should be 2'
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = data[:,0]
    Y = data[:,1]
    bins = 10
    hist, xedges, yedges = np.histogram2d(X, Y, bins=bins, range=[[0, bins*0.6], [0, bins*0.6]])

    # Construct arrays for the anchor positions of the bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = hist.ravel()

    cmap = cm.get_cmap('cool')
    max_height = np.max(dz)
    min_height = np.min(dz)
    rgba = [cmap((k-min_height)/max_height) for k in dz] 
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', color=rgba)

    plt.show()

def plot_3d_surface(data, func):
    cols = data.shape[1]
    if cols < 2:
        assert False, 'The number of columns should be 2'
    X = data[:,0]
    Y = data[:,1]
    X, Y = np.meshgrid(X, Y)
    Z = func(X, Y)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('surface');

def plot_3d_scatter(data, title=None, xlabel=None, ylabel=None, zlabel=None):
    cols = data.shape[1]
    if cols < 3:
        assert False, 'The number of columns should be 3'
    X = data[:,0]
    Y = data[:,1]
    Z = data[:,2]
    ax = plt.axes(projection='3d')
    ax.scatter(X, Y, Z, c = Z, cmap='RdYlGn')
    ax.set_title(title);
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.show()

def plot_3d_scatter_trisurf(data, title=None, xlabel=None, ylabel=None, zlabel=None):
    cols = data.shape[1]
    if cols < 3:
        assert False, 'The number of columns should be 3'
    X = data[:,0]
    Y = data[:,1]
    Z = data[:,2]

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title);
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    #ax.plot_surface(XX, YY, ZZ)
    surf = ax.plot_trisurf(X - X.mean(), Y - Y.mean(), Z - Z.mean(), cmap=cm.jet, linewidth=0.1)
    fig.colorbar(surf)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.zaxis.set_major_locator(MaxNLocator(6))
    fig.tight_layout()
    plt.show()

def plot_3d_scatter_surface(data, title=None, xlabel=None, ylabel=None, zlabel=None):
    plot_3d_scatter_kde(data, title, xlabel, ylabel, zlabel)

def plot_3d_scatter_fxy(data, title=None, xlabel=None, ylabel=None, zlabel=None):
    cols = data.shape[1]
    if cols < 2:
        assert False, 'The number of columns should be 2'
    X = data[:,0]
    Y = data[:,1]
    XY = np.vstack([X,Y])
    Z = gaussian_kde(XY)(XY)

    # Sort the points by density, so that the densest points are plotted last
    idx = Z.argsort()
    x, y, z = X[idx], Y[idx], Z[idx]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #surf = ax.plot_trisurf(x - x.mean(), y - y.mean(), z, cmap=cm.jet, linewidth=0.1)
    surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.1)
    ax.scatter(x,y,z, marker='.', s=10, c="black", alpha=0.5)
    fig.colorbar(surf)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.zaxis.set_major_locator(MaxNLocator(6))
    fig.tight_layout()


    ax.set_title(title);
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.show()

    return

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title);
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.plot_surface(XX, YY, ZZ)
    plt.show()

def plot_3d_scatter_kde(data, title=None, xlabel=None, ylabel=None, zlabel=None):
    cols = data.shape[1]
    if cols < 2:
        assert False, 'The number of columns should be 2'
    X = data[:,0]
    Y = data[:,1]

    Xmin = int(np.floor(np.amin(X)))
    Xmax = int(np.ceil(np.amax(X)))
    Ymin = int(np.floor(np.amin(Y)))
    Ymax = int(np.ceil(np.amax(Y)))

    xmin = min(Xmin, Ymin)
    ymin = min(Xmin, Ymin)
    xmax = max(Xmax, Ymax)
    ymax = max(Xmax, Ymax)

    # Peform the kernel density estimate
    xx, yy = np.mgrid[Xmin:Xmax:100j, Ymin:Ymax:100j]
    #xx, yy = np.meshgrid(X, Y)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([X, Y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, f - f.mean(), rstride=1, cstride=1, cmap='jet', edgecolor='none')
    
    ## Or kernel density estimate plot instead of the contourf plot
    cset = ax.contour(xx, yy, f, colors='k')
    # Label plot
    ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_title(title)
    plt.show()

def plot_2d_scatter(data, title=None, xlabel=None, ylabel=None, handles=None):
    cols = data.shape[1]
    if cols < 3:
        assert False, 'The number of columns should be 3'
    X = data[:,0]
    Y = data[:,1]
    Z = data[:,2]
    ax = plt.axes()
    scatter = ax.scatter(X, Y, c = ['green' if z > 0.5 else 'red' for z in Z], cmap='RdYlGn')
    ax.set_title(title);
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.legend(handles=handles)
    plt.show()     


def main():
    for i in range(9):
        device_index = i
        device_id = device_index + 1
        device_data = get_nbaiot_device_data(device_id)
        plot_surface_nbaiot_device(device_data, device_id)
        plot_scatter_nbaiot_device(device_data, device_id, False)  
        

if __name__ == "__main__":
    main()