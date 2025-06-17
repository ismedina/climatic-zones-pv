import numpy as np
from mpl_toolkits.basemap import Basemap
import ot
import reverse_geocoder


def compute_R2(x, y, weights=None):
    """
    Weighted R2 correlation coefficient for best linear fit.
    """
    x = np.array(x)
    y = np.array(y)
    if weights is None:
        c0, c1 = np.polyfit(x, y, 1)
        error = np.sum((y - c0*x - c1)**2)
        error_mean = np.sum((y - np.mean(y))**2)
    else:
        A = np.hstack((x.reshape(-1, 1), np.ones((len(x), 1))))
        c0, c1 = np.linalg.solve((A.T * weights) @ A, A.T @ (y * weights))
        error = np.sum(weights * (y - c0*x - c1)**2)
        meany = np.sum(weights * y) / np.sum(weights)
        error_mean = np.sum(weights * (y - meany)**2)
    return (1 - error/error_mean)


# Adapted from a keops tutorial
def KMeans(x, K=10, Niter=20, index_init=None, weights=None, p=2):
    """
    Implements Lloyd's algorithm for the Euclidean metric.
    """
    N, D = x.shape  # Number of samples, dimension of the ambient space

    # index_init gives initialization for
    if index_init is None:
        index_init = slice(None, None, (N//K+1))
    else:
        index_init = np.array(index_init)
    if weights is None:
        weights = np.ones(N)
    c = x[index_init, :].copy()  # Initialization for the centroids
    x_i = x.reshape(N, 1, D)  # (N, 1, D) samples
    c_j = c.reshape(1, K, D)  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for _ in range(Niter):
        # E step: assign points to the closest cluster -------------------------
        if p == np.inf:
            D_ij = np.abs(x_i - c_j).max(-1)
        else:
            D_ij = (np.abs(x_i - c_j) ** p).sum(-1)  # (N, K) Lp distances
        cl = D_ij.argmin(axis=1).reshape(-1)  # Points -> Nearest cluster
        # Update the centroids to the normalized cluster average
        for i in range(K):
            x_cluster = x[cl == i, :]
            w_cluster = weights[cl == i].reshape(-1, 1)
            c[i] = np.sum(w_cluster * x_cluster, axis=0) / np.sum(w_cluster)

    # Find representants for cluster within the dataset----------
    if p == np.inf:
        D_ij = np.abs(x_i - c_j).max(-1)
    else:
        D_ij = (np.abs(x_i - c_j) ** p).sum(-1)  # (N, K) Lp distances
    index_closest_c = D_ij.argmin(axis=0)
    return cl, c, index_closest_c


def drawmap(ax, coords, projection="cass", step_grid=5, label_coords=None,
            basemap_res="l"):
    """
    Draw map using matplotlib Basemap.
    """
    # llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
    # are the lat/lon values of the lower left and upper right corners
    # of the map.
    # resolution = 'i' means use intermediate resolution coastlines.
    # lon_0, lat_0 are the central longitude and latitude of the projection.
    lat0, lat1, lon0, lon1 = coords
    # if lat0 == "-180":  # Full map
    #     basemap_res = "l"
    # else:  # Europe
    #     basemap_res = "i"

    m = Basemap(llcrnrlon=lon0, llcrnrlat=lat0, urcrnrlon=lon1, urcrnrlat=lat1,
                resolution=basemap_res,  # "c, l, i, h, f"
                projection=projection,
                lon_0=(lon0+lon1)/2, lat_0=(lat0+lat1)/2,
                ax=ax)
    # can get the identical map this way (by specifying width and
    # height instead of lat/lon corners)
    # m = Basemap(width=891185,height=1115557,\
    #            resolution='i',projection='cass',lon_0=-4.36,lat_0=54.7)
    # m.drawcoastlines(linewidth = 0.25)
    m.fillcontinents(color='lightgray', lake_color=(0.9, 0.9, 0.9))
    # draw parallels and meridians.
    m.drawcountries(linewidth=0.25, zorder=100)
    n0, n1, m0, m1 = np.array([lat0, lat1, lon0, lon1]) // step_grid
    if label_coords is None:
        label_coords = [True, False, False, True]
    m.drawparallels(np.arange(n0, n1+1)*step_grid,
                    labels=label_coords, zorder=99, linewidth=0.5)
    m.drawmeridians(np.arange(m0, m1+1)*step_grid,
                    labels=label_coords, zorder=99, linewidth=0.5)
    m.drawmapboundary(fill_color=(0.95, 0.95, 0.95))
    return m


def is_in_region(lats, lons, region_codes):
    """
    Checks where the coordinates `lats, lons` are in the ISO regions 
    corresponding to `region_codes`.
    """
    tuples_lats_lons = list(zip(lats, lons))
    results = reverse_geocoder.search(tuples_lats_lons)
    return [res["cc"] in region_codes for res in results]


def module_temperature(G, T, ws, u0=26.91, u1=6.2):
    """
    Computes PV module temperature based on irradiance, ambient temperature and
    windspeed. Coefficients are taking from [1].

    References
    ==========
    [1] T. Huld and A. M. Gracia Amillo, “Estimating PV module performance over
    large geographical regions: The role of irradiance, air temperature, wind 
    speed and solar spectrum,” Energies, vol. 8, no. 6, pp. 5159–5181, 2015, 
    doi: 10.3390/en8065159.
    """
    return T + G/(u0 + u1*ws)


def power_king(G, Tmod, ks=[-0.006756, -0.016444, -0.003015, -0.000045,
                            -0.000043, 0.0]):
    """
    Computes PV power based on G and Tmod, using the linearized king model [1].
    The default coefficients are taken from [2].

    References
    ==========
    [1] T. Huld and A. M. Gracia Amillo, “Estimating PV module performance over
    large geographical regions: The role of irradiance, air temperature, wind 
    speed and solar spectrum,” Energies, vol. 8, no. 6, pp. 5159–5181, 2015, 
    doi: 10.3390/en8065159.
    [2] A. Chatzipanagi, N. Taylor, I. Medina, T. Lyubenova, A. Martinez 
    and E. D. Dunlop, “An Updated Simplified Energy Yield Model for Recent 
    Photovoltaic Module Technologies,”, in preparation.
    """
    Gp = G / 1000
    Tp = Tmod - 25
    mask0 = Gp <= 0
    Gp[mask0] = 1e-5  # To avoid warning in logarithm
    logGp = np.log(Gp)
    k1, k2, k3, k4, k5, k6 = ks
    eta = 1 + k1*logGp + k2*logGp**2 + k3*Tp \
        + k4*Tp*logGp + k5*Tp*logGp**2 + k6*Tp**2
    eta[mask0] = 0.0
    P = G * eta
    return P


def rasterize(lats, lons, z):
    """
    Turn point cloud `z` supported on `(lats, lons)` into an image.
    """
    lats_unique = np.unique(np.array(lats))
    dlat = np.min(np.diff(np.sort(lats_unique)))
    min_lat, max_lat = np.min(lats_unique), np.max(lats_unique)
    min_lon, max_lon = np.min(lons), np.max(lons)
    shape_img = int((max_lat - min_lat) / dlat) + 1, \
        int((max_lon - min_lon) / dlat) + 1
    idx_lat = np.array(np.round((lats - min_lat) / dlat), dtype=np.int64)
    idx_lon = np.array(np.round((lons - min_lon) / dlat), dtype=np.int64)
    # Get weights according to longitude
    lat_weights = np.array(np.cos(2*np.pi*lats / 360))
    img = np.ones(shape_img) * np.nan
    img[idx_lat, idx_lon] = z
    extent = (min_lon-dlat/2, max_lon+dlat/2, min_lat-dlat/2, max_lat+dlat/2)
    return img, extent


def random_representatives(z, c, tol=0.01):
    """
    For each row in `c`, sample a row from `z` such that the distance between
    them is less than `tol`. Returns the index of the sampled rows. 
    """
    Ncomp = z.shape[-1]
    D_ij = np.sqrt(
        ((z.reshape(-1, 1, Ncomp) - c.reshape(1, -1, Ncomp)) ** 2).sum(-1))
    ind = np.zeros(len(c))
    total_reps = 0
    for i in range(len(c)):
        ind_ = np.where(D_ij[:, i] < tol)[0]
        total_reps += len(ind_)
        ind[i] = np.random.choice(ind_)
    print("Total representatives", total_reps)
    return ind


def permute_centroid(z_ref, z, ind):
    """
    Permutes the rows of `z` so that the corresponding rows from `z_ref` are 
    as close as possible, in squared distance.
    """
    N, M = len(z_ref), len(z)
    C = ((z_ref[:, None, :] - z[None, :, :])**2).sum(-1)
    mu, nu = np.ones(N)/N, np.ones(M)/M
    gamma = ot.emd(mu, nu, C)
    # Get ordering for z
    order = np.arange(N) @ gamma
    order = np.argsort(order)
    return z[order], ind[order]


def subsample_df(df, res):
    """
    Subsample geospatial `df` according to resolution `res`. Assumes that `res`
    is a multiple of the resolution of `df`.
    """
    ind_lat = np.round(df.lat / res, 3) % 1 == 0
    ind_lon = np.round(df.lon / res, 3) % 1 == 0
    return df[ind_lat & ind_lon]
