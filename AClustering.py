# I would like to acknowledge the use of OpenAI's ChatGPT as a tool to support the development of this report. 
# It assisted in refining code, and clarifying technical explanations related to data processing and model interpretation.

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

def elbow():
    # Load CSVs (adjust paths if needed)
    accident_df = pd.read_csv("accident.csv")
    vehicle_df = pd.read_csv("filtered_vehicle.csv")

    # Merge on ACCIDENT_NO
    df = pd.merge(accident_df, vehicle_df, on="ACCIDENT_NO", how="inner")

    # Convert accident time to hour
    df['HOUR'] = pd.to_datetime(df['ACCIDENT_TIME'], format='%H:%M:%S', errors='coerce').dt.hour

    # Remove significant speed zone outliers
    dfc = df[df['SPEED_ZONE']!=999]

    # Group by key identifiers
    df_group = dfc.groupby(['VEHICLE_TYPE_DESC', 'TRAFFIC_CONTROL_DESC', 'ROAD_GEOMETRY_DESC']).agg({
        'HOUR': 'mean',
        'SPEED_ZONE': 'mean',
        'SEATING_CAPACITY': 'mean',
        'TARE_WEIGHT': 'mean'
    }).reset_index()

    # Select features for clustering
    features = ['HOUR', 'SPEED_ZONE', 'SEATING_CAPACITY', 'TARE_WEIGHT']

    # Normalize features using MinMaxScaler
    scaler = MinMaxScaler()
    df_group[features] = scaler.fit_transform(df_group[features]) 

    # Elbow method to find optimal k
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=10, n_init=10)
        kmeans.fit(df_group[features])
        sse.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), sse, marker='o', linestyle='-')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Sum of Squared Errors (SSE)")
    plt.title("Elbow Method For Optimal K", fontsize=16, fontweight='bold', pad=20)

    plt.show()

    return

elbow()

###### the above is the elbow methods, below is the actual clustering

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

def cluster():
    # Load data
    accident_df = pd.read_csv("accident.csv")
    vehicle_df = pd.read_csv("filtered_vehicle.csv")

    # Merge datasets
    df = pd.merge(accident_df, vehicle_df, on="ACCIDENT_NO", how="inner")

    # Convert accident time to hour of day using specified format
    df['HOUR'] = pd.to_datetime(df['ACCIDENT_TIME'], format='%H:%M:%S', errors='coerce').dt.hour

    # Remove significant speed zone outliers
    dfc = df[df['SPEED_ZONE']!=999]

    # Group data for clustering
    df_group = dfc.groupby(['VEHICLE_TYPE_DESC', 'TRAFFIC_CONTROL_DESC', 'ROAD_GEOMETRY_DESC']).agg({
        'HOUR': 'mean',
        'SPEED_ZONE': 'mean',
        'SEATING_CAPACITY': 'mean',
        'TARE_WEIGHT': 'mean'
    }).reset_index()

    # Feature selection
    features = ['HOUR', 'SPEED_ZONE', 'SEATING_CAPACITY', 'TARE_WEIGHT']
    scaler = MinMaxScaler()
    df_group_scaled = df_group.copy()
    df_group_scaled[features] = scaler.fit_transform(df_group_scaled[features])

    # KMeans clustering with k = 3
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=10, n_init=10)
    df_group_scaled['Cluster'] = kmeans.fit_predict(df_group_scaled[features])

    # Plot the clusters
    import seaborn as sns

    import matplotlib.pyplot as plt

    # Basic color map for up to 3 clusters
    colors = ['blue', 'green', 'red']
    cluster_colors = [colors[i] for i in df_group_scaled['Cluster']]

# Create the plot
    plt.figure(figsize=(10, 6))
    for cluster_id in sorted(df_group_scaled['Cluster'].unique()):
        cluster_data = df_group_scaled[df_group_scaled['Cluster'] == cluster_id]
        plt.scatter(
            cluster_data['HOUR'],
            cluster_data['SPEED_ZONE'],
            color=colors[cluster_id],
            label=f'Cluster {cluster_id}',
            edgecolors='k',
            alpha=0.7,
            s=50
        )

    # Labels and title
    plt.yscale('log')
    plt.xlabel('Hour of Accident (scaled)')
    plt.ylabel('Speed Zone (scaled)')
    plt.title('Crashes by Hour and Speed Zone with K-Means Clusters (k=3)', fontsize=14, fontweight='bold')
    plt.legend(title='Clusters')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.show()
    
    for i in sorted(df_group_scaled['Cluster'].unique()):
        cluster_df = df_group_scaled[df_group_scaled['Cluster'] == i]
        filename = f'task3_3_cluster_{i}.csv'
        cluster_df.to_csv(filename, index=False)
        print(f"Cluster {i} saved to '{filename}'")

    return

cluster()