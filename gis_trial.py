import pandas as pd
import networkx as nx
import geojson
from pyspark.sql import SparkSession
from shapely.geometry import Point, LineString
from collections import deque
import numpy as np
import geopandas as gpd
import community

# Initialize Spark Session
spark = SparkSession.builder.appName("RoadNetworkAnalysis").getOrCreate()
sc = spark.sparkContext

def compute_betweenness(nodes_file, edges_file, output_geojson):
    # Read node coordinates
    nodes_rdd = spark.read.text(nodes_file).rdd
    nodes_rdd = nodes_rdd.zipWithIndex().filter(lambda x: x[1] > 0)  # Skip first line

    nodes_list = [
        (float(parts[0]), float(parts[1]))
        for line, _ in nodes_rdd.collect()
        if (parts := line.value.strip().split()) and len(parts) == 2 and not line.value.startswith("#")
    ]
    nodes_df = pd.DataFrame(nodes_list, columns=["latitude", "longitude"])
    nodes_df["nodeID"] = nodes_df.index

    # Read edges (distance)
    edges_distance_rdd = spark.read.text(edges_file).rdd
    edges_list = [
        (int(parts[0]), int(parts[1]), float(parts[2]))
        for line in edges_distance_rdd.collect()
        if (parts := line.value.strip().split()) and len(parts) == 3 and not line.value.startswith("#")
    ]
    edges_distance_df = pd.DataFrame(edges_list, columns=["source", "destination", "distance"])

    # Define Smaller Region (Central Business District)
    region_lat_min, region_lat_max = 22.27, 22.29
    region_lon_min, region_lon_max = 114.15, 114.18

    # Filter nodes based on the defined region
    region_nodes = nodes_df[
        (nodes_df['latitude'] >= region_lat_min) & (nodes_df['latitude'] <= region_lat_max) & 
        (nodes_df['longitude'] >= region_lon_min) & (nodes_df['longitude'] <= region_lon_max)
    ]
    region_node_ids = set(region_nodes['nodeID'].values)

    # Filter edges to include only those within the region
    filtered_edges = edges_distance_df[
        (edges_distance_df['source'].isin(region_node_ids)) & (edges_distance_df['destination'].isin(region_node_ids))
    ]

    # Create NetworkX Graph
    G = nx.Graph()
    for _, row in filtered_edges.iterrows():
        G.add_edge(row["source"], row["destination"], weight=row["distance"])

    # Custom Betweenness Centrality Calculation
    def custom_betweenness_centrality(graph):
        betweenness = {node: 0.0 for node in graph.nodes()}
        nodes = list(graph.nodes())
        num_nodes = len(nodes)

        for i, source in enumerate(nodes):
            shortest_paths = {node: [] for node in nodes}
            shortest_paths[source] = [[source]]
            queue = [source]
            while queue:
                current = queue.pop(0)
                for neighbor in graph.neighbors(current):
                    if not shortest_paths[neighbor]:
                        queue.append(neighbor)
                        shortest_paths[neighbor] = [path + [neighbor] for path in shortest_paths[current]]
                    elif len(shortest_paths[neighbor][0]) == len(shortest_paths[current][0]) + 1:
                        shortest_paths[neighbor].extend([path + [neighbor] for path in shortest_paths[current]])
            
            dependency = {node: 0 for node in nodes}
            for target in reversed(nodes):
                if target != source:
                    for path in shortest_paths[target]:
                        for node in path[:-1]:
                            dependency[node] += 1 / len(shortest_paths[target])
                            betweenness[node] += dependency[node]
            
            print(f"Processed {i+1}/{num_nodes} nodes")
        return betweenness

    betweenness_centrality = custom_betweenness_centrality(G)

    # Convert betweenness centrality into a DataFrame
    betweenness_df = pd.DataFrame(list(betweenness_centrality.items()), columns=["nodeID", "Betweenness Centrality"])

    # Merge with node coordinates
    betweenness_df = betweenness_df.merge(region_nodes, on="nodeID")

    # Prepare GeoJSON data structure
    features = []
    for _, row in betweenness_df.iterrows():
        point = Point((row['longitude'], row['latitude']))
        feature = {
            "type": "Feature",
            "geometry": point.__geo_interface__,
            "properties": {
                "NodeID": row["nodeID"],
                "Betweenness Centrality": row["Betweenness Centrality"]
            }
        }
        features.append(feature)

    # Create GeoJSON
    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }

    # Save to a GeoJSON file
    with open(output_geojson, "w") as f:
        geojson.dump(geojson_data, f)

    print(f"GeoJSON file saved as '{output_geojson}'")

    # Print how many nodes were processed
    print(f"Processed {len(region_nodes)} nodes for betweenness centrality.")


def compute_closeness(nodes_file, edges_file, output_geojson):
    # Read node coordinates
    nodes_rdd = spark.read.text(nodes_file).rdd
    nodes_rdd = nodes_rdd.zipWithIndex().filter(lambda x: x[1] > 0)  # Skip first line

    nodes_list = [
        (float(parts[0]), float(parts[1]))
        for line, _ in nodes_rdd.collect()
        if (parts := line.value.strip().split()) and len(parts) == 2 and not line.value.startswith("#")
    ]
    nodes_df = pd.DataFrame(nodes_list, columns=["latitude", "longitude"])
    nodes_df["nodeID"] = nodes_df.index

    # Read edges (distance)
    edges_distance_rdd = spark.read.text(edges_file).rdd
    edges_list = [
        (int(parts[0]), int(parts[1]), float(parts[2]))
        for line in edges_distance_rdd.collect()
        if (parts := line.value.strip().split()) and len(parts) == 3 and not line.value.startswith("#")
    ]
    edges_distance_df = pd.DataFrame(edges_list, columns=["source", "destination", "distance"])

    # Define Smaller Region (Central Business District)
    region_lat_min, region_lat_max = 22.27, 22.29
    region_lon_min, region_lon_max = 114.15, 114.18

    # Filter nodes based on the defined region
    region_nodes = nodes_df[
        (nodes_df['latitude'] >= region_lat_min) & (nodes_df['latitude'] <= region_lat_max) & 
        (nodes_df['longitude'] >= region_lon_min) & (nodes_df['longitude'] <= region_lon_max)
    ]
    region_node_ids = set(region_nodes['nodeID'].values)

    # Filter edges to include only those within the region
    filtered_edges = edges_distance_df[
        (edges_distance_df['source'].isin(region_node_ids)) & (edges_distance_df['destination'].isin(region_node_ids))
    ]

    # Create NetworkX Graph
    G = nx.Graph()
    for _, row in filtered_edges.iterrows():
        G.add_edge(row["source"], row["destination"], weight=row["distance"])

    # Compute closeness centrality
    closeness_centrality = nx.closeness_centrality(G)

    # Add closeness centrality to the DataFrame
    region_nodes["Closeness Centrality"] = region_nodes["nodeID"].map(closeness_centrality)

    # Replace NaN values with 0 (or another default value of your choice)
    region_nodes["Closeness Centrality"].fillna(0, inplace=True)

    # Prepare GeoJSON data structure
    features = []
    for _, row in region_nodes.iterrows():
        point = Point((row['longitude'], row['latitude']))
        feature = {
            "type": "Feature",
            "geometry": point.__geo_interface__,
            "properties": {
                "NodeID": row["nodeID"],
                "Closeness Centrality": row["Closeness Centrality"]
            }
        }
        features.append(feature)

    # Create GeoJSON
    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }

    # Save to a GeoJSON file
    with open(output_geojson, "w") as f:
        geojson.dump(geojson_data, f)

    print(f"GeoJSON file saved as '{output_geojson}'")

    # Print how many nodes were processed
    print(f"Processed {len(region_nodes)} nodes for closeness centrality.")


def bfs_shortest_paths(graph, start_node):
    """
    Perform BFS to compute shortest paths from start_node to all other nodes in the graph.
    """
    distances = {node: float('inf') for node in graph.nodes()}
    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        node = queue.popleft()
        current_distance = distances[node]
        
        for neighbor in graph.neighbors(node):
            if distances[neighbor] == float('inf'):  # Node has not been visited yet
                distances[neighbor] = current_distance + 1
                queue.append(neighbor)

    return distances

def compute_closeness(nodes_file, edges_file, output_geojson):
    # Read node coordinates
    nodes_rdd = spark.read.text(nodes_file).rdd
    nodes_rdd = nodes_rdd.zipWithIndex().filter(lambda x: x[1] > 0)  # Skip first line

    nodes_list = [
        (float(parts[0]), float(parts[1]))
        for line, _ in nodes_rdd.collect()
        if (parts := line.value.strip().split()) and len(parts) == 2 and not line.value.startswith("#")
    ]
    nodes_df = pd.DataFrame(nodes_list, columns=["latitude", "longitude"])
    nodes_df["nodeID"] = nodes_df.index

    # Read edges (distance)
    edges_distance_rdd = spark.read.text(edges_file).rdd
    edges_list = [
        (int(parts[0]), int(parts[1]), float(parts[2]))
        for line in edges_distance_rdd.collect()
        if (parts := line.value.strip().split()) and len(parts) == 3 and not line.value.startswith("#")
    ]
    edges_distance_df = pd.DataFrame(edges_list, columns=["source", "destination", "distance"])

    # Define Smaller Region (Central Business District)
    region_lat_min, region_lat_max = 22.27, 22.29
    region_lon_min, region_lon_max = 114.15, 114.18

    # Filter nodes based on the defined region
    region_nodes = nodes_df[
        (nodes_df['latitude'] >= region_lat_min) & (nodes_df['latitude'] <= region_lat_max) & 
        (nodes_df['longitude'] >= region_lon_min) & (nodes_df['longitude'] <= region_lon_max)
    ]
    region_node_ids = set(region_nodes['nodeID'].values)

    # Filter edges to include only those within the region
    filtered_edges = edges_distance_df[
        (edges_distance_df['source'].isin(region_node_ids)) & (edges_distance_df['destination'].isin(region_node_ids))
    ]

    # Create NetworkX Graph
    G = nx.Graph()
    for _, row in filtered_edges.iterrows():
        G.add_edge(row["source"], row["destination"], weight=row["distance"])

    # Custom Closeness Centrality Calculation
    def custom_closeness_centrality(graph):
        closeness = {}
        nodes = list(graph.nodes())
        num_nodes = len(nodes)

        for i, node in enumerate(nodes):
            distances = bfs_shortest_paths(graph, node)
            total_distance = sum(distances.values())
            
            if total_distance > 0:  # Avoid division by zero
                closeness[node] = (num_nodes - 1) / total_distance
            else:
                closeness[node] = 0
            
            print(f"Processed {i+1}/{num_nodes} nodes")

        return closeness

    # Compute the closeness centrality for each node
    closeness_centrality = custom_closeness_centrality(G)

    # Add closeness centrality to the DataFrame using .loc to avoid the SettingWithCopyWarning
    region_nodes.loc[:, "Closeness Centrality"] = region_nodes["nodeID"].map(closeness_centrality)

    # Replace NaN values with 0 (or another default value of your choice)
    region_nodes.loc[:, "Closeness Centrality"].fillna(0, inplace=True)

    # Prepare GeoJSON data structure
    features = []
    for _, row in region_nodes.iterrows():
        point = Point((row['longitude'], row['latitude']))
        feature = {
            "type": "Feature",
            "geometry": point.__geo_interface__,
            "properties": {
                "NodeID": row["nodeID"],
                "Closeness Centrality": row["Closeness Centrality"]
            }
        }
        features.append(feature)

    # Create GeoJSON
    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }

    # Save to a GeoJSON file
    with open(output_geojson, "w") as f:
        geojson.dump(geojson_data, f)

    print(f"GeoJSON file saved as '{output_geojson}'")

    # Print how many nodes were processed
    print(f"Processed {len(region_nodes)} nodes for closeness centrality.")



def compute_degree_centrality(nodes_file, edges_file, output_geojson, output_nodes_geojson, output_edges_geojson):
    # Read node coordinates
    nodes_rdd = spark.read.text(nodes_file).rdd
    nodes_rdd = nodes_rdd.zipWithIndex().filter(lambda x: x[1] > 0)  # Skip first line

    nodes_list = [
        (float(parts[0]), float(parts[1]))
        for line, _ in nodes_rdd.collect()
        if (parts := line.value.strip().split()) and len(parts) == 2 and not line.value.startswith("#")
    ]
    nodes_df = pd.DataFrame(nodes_list, columns=["latitude", "longitude"])
    nodes_df["nodeID"] = nodes_df.index

    # Define region (popular region)
    region_lat_min, region_lat_max = 22.27, 22.29
    region_lon_min, region_lon_max = 114.15, 114.18

    # Filter nodes based on the region
    region_nodes = nodes_df[
        (nodes_df['latitude'] >= region_lat_min) & (nodes_df['latitude'] <= region_lat_max) &
        (nodes_df['longitude'] >= region_lon_min) & (nodes_df['longitude'] <= region_lon_max)
    ]
    region_node_ids = set(region_nodes['nodeID'].values)

    # Read edges (distance)
    edges_distance_rdd = spark.read.text(edges_file).rdd
    edges_list = [
        (int(parts[0]), int(parts[1]), float(parts[2]))
        for line in edges_distance_rdd.collect()
        if (parts := line.value.strip().split()) and len(parts) == 3 and not line.value.startswith("#")
    ]
    edges_distance_df = pd.DataFrame(edges_list, columns=["source", "destination", "distance"])

    # Filter edges to include only those within the region
    filtered_edges = edges_distance_df[
        (edges_distance_df['source'].isin(region_node_ids)) & (edges_distance_df['destination'].isin(region_node_ids))
    ]

    # Create NetworkX Graph
    G = nx.DiGraph()  # Directed graph for degree centrality
    for _, row in filtered_edges.iterrows():
        G.add_edge(row["source"], row["destination"], weight=row["distance"])

    # Degree Centrality Calculation
    def compute_degree_centrality(graph):
        # Calculate the degree for each node
        degree_centrality = {node: graph.degree(node) for node in graph.nodes()}

        # Normalize the degree centrality by dividing each node's degree by the maximum degree
        max_degree = max(degree_centrality.values())
        normalized_centrality = {node: degree / max_degree for node, degree in degree_centrality.items()}

        return normalized_centrality

    # Compute Degree Centrality
    degree_centrality_values = compute_degree_centrality(G)

    # Merge with node coordinates for GeoJSON output
    degree_centrality_df = pd.DataFrame(list(degree_centrality_values.items()), columns=["nodeID", "Degree Centrality"])
    degree_centrality_df = degree_centrality_df.merge(region_nodes, on="nodeID")

    # Prepare GeoJSON data for nodes
    node_features = []
    for _, row in degree_centrality_df.iterrows():
        point = Point((row['longitude'], row['latitude']))
        feature = {
            "type": "Feature",
            "geometry": point.__geo_interface__,
            "properties": {
                "NodeID": row["nodeID"],
                "Latitude": row["latitude"],
                "Longitude": row["longitude"]
            }
        }
        node_features.append(feature)

    # Create Node GeoJSON
    node_geojson_data = {
        "type": "FeatureCollection",
        "features": node_features
    }

    # Save Node GeoJSON
    with open(output_nodes_geojson, "w") as f:
        geojson.dump(node_geojson_data, f)
    print(f"Node GeoJSON file saved as '{output_nodes_geojson}'")

    # Prepare GeoJSON data for edges
    edge_features = []
    for _, row in filtered_edges.iterrows():
        source_node = row["source"]
        dest_node = row["destination"]
        source_coords = degree_centrality_df[degree_centrality_df['nodeID'] == source_node][['longitude', 'latitude']].values[0]
        dest_coords = degree_centrality_df[degree_centrality_df['nodeID'] == dest_node][['longitude', 'latitude']].values[0]
        
        line = LineString([source_coords[::-1], dest_coords[::-1]])  # Reverse to (longitude, latitude)
        feature = {
            "type": "Feature",
            "geometry": line.__geo_interface__,
            "properties": {
                "Source": source_node,
                "Destination": dest_node,
                "Distance": row["distance"]
            }
        }
        edge_features.append(feature)

    # Create Edge GeoJSON
    edge_geojson_data = {
        "type": "FeatureCollection",
        "features": edge_features
    }

    # Save Edge GeoJSON
    with open(output_edges_geojson, "w") as f:
        geojson.dump(edge_geojson_data, f)
    print(f"Edge GeoJSON file saved as '{output_edges_geojson}'")

    # Prepare GeoJSON data for Degree Centrality
    degree_centrality_features = []
    for _, row in degree_centrality_df.iterrows():
        point = Point((row['longitude'], row['latitude']))
        feature = {
            "type": "Feature",
            "geometry": point.__geo_interface__,
            "properties": {
                "NodeID": row["nodeID"],
                "Degree Centrality": row["Degree Centrality"]
            }
        }
        degree_centrality_features.append(feature)

    # Create Degree Centrality GeoJSON
    degree_centrality_geojson_data = {
        "type": "FeatureCollection",
        "features": degree_centrality_features
    }

    # Save Degree Centrality GeoJSON
    with open(output_geojson, "w") as f:
        geojson.dump(degree_centrality_geojson_data, f)

    print(f"Degree Centrality GeoJSON file saved as '{output_geojson}'")





def create_nodes_edges_geojson(nodes_file, edges_file, output_nodes_geojson, output_edges_geojson):
    # Read node coordinates
    nodes_rdd = spark.read.text(nodes_file).rdd
    nodes_rdd = nodes_rdd.zipWithIndex().filter(lambda x: x[1] > 0)  # Skip first line

    nodes_list = [
        (float(parts[0]), float(parts[1]))
        for line, _ in nodes_rdd.collect()
        if (parts := line.value.strip().split()) and len(parts) == 2 and not line.value.startswith("#")
    ]
    nodes_df = pd.DataFrame(nodes_list, columns=["latitude", "longitude"])
    nodes_df["nodeID"] = nodes_df.index

    # Read edges (distance)
    edges_distance_rdd = spark.read.text(edges_file).rdd
    edges_list = [
        (int(parts[0]), int(parts[1]), float(parts[2]))
        for line in edges_distance_rdd.collect()
        if (parts := line.value.strip().split()) and len(parts) == 3 and not line.value.startswith("#")
    ]
    edges_distance_df = pd.DataFrame(edges_list, columns=["source", "destination", "distance"])

    # Prepare GeoJSON data for nodes
    node_features = []
    for _, row in nodes_df.iterrows():
        point = Point((row['longitude'], row['latitude']))  # Correct order: (longitude, latitude)
        feature = {
            "type": "Feature",
            "geometry": point.__geo_interface__,
            "properties": {
                "NodeID": row["nodeID"],
                "Latitude": row["latitude"],
                "Longitude": row["longitude"]
            }
        }
        node_features.append(feature)

    # Create Node GeoJSON
    node_geojson_data = {
        "type": "FeatureCollection",
        "features": node_features
    }

    # Save Node GeoJSON
    with open(output_nodes_geojson, "w") as f:
        geojson.dump(node_geojson_data, f)
    print(f"Node GeoJSON file saved as '{output_nodes_geojson}'")

    # Prepare GeoJSON data for edges
    edge_features = []
    for _, row in edges_distance_df.iterrows():
        source_node = row["source"]
        dest_node = row["destination"]

        # Get source and destination coordinates
        source_coords = nodes_df[nodes_df['nodeID'] == source_node][['longitude', 'latitude']].values
        dest_coords = nodes_df[nodes_df['nodeID'] == dest_node][['longitude', 'latitude']].values

        # If the coordinates are found, create the edge line
        if source_coords.size > 0 and dest_coords.size > 0:
            source_coords = source_coords[0]
            dest_coords = dest_coords[0]

            line = LineString([source_coords[::-1], dest_coords[::-1]])  # Reverse to (longitude, latitude)
            feature = {
                "type": "Feature",
                "geometry": line.__geo_interface__,
                "properties": {
                    "Source": source_node,
                    "Destination": dest_node,
                    "Distance": row["distance"]
                }
            }
            edge_features.append(feature)

    # Create Edge GeoJSON
    edge_geojson_data = {
        "type": "FeatureCollection",
        "features": edge_features
    }

    # Save Edge GeoJSON
    with open(output_edges_geojson, "w") as f:
        geojson.dump(edge_geojson_data, f)
    print(f"Edge GeoJSON file saved as '{output_edges_geojson}'")



# File paths
nodes_file = "hdfs://spark-yarn-master:8080/data/Hongkong/Hongkong.co"
edges_distance_file = "hdfs://spark-yarn-master:8080/data/Hongkong/Hongkong.road-d"
edges_travel_file = "hdfs://spark-yarn-master:8080/data/Hongkong/Hongkong.road-t"

def create_nodes_and_edges_geojson(nodes_file, edges_distance_file, edges_travel_file, output_nodes_geojson, output_edges_geojson):
    # Read node coordinates (Skipping first line)
    nodes_rdd = spark.read.text(nodes_file).rdd
    nodes_rdd = nodes_rdd.zipWithIndex().filter(lambda x: x[1] > 0)  # Skip first line

    nodes_list = []
    for line, _ in nodes_rdd.collect():
        parts = line.value.strip().split()
        if len(parts) == 2:
            try:
                lat, lon = float(parts[0]), float(parts[1])
                nodes_list.append((lat, lon))
            except ValueError:
                print(f"Skipping invalid line: {line.value.strip()}")
    nodes_rdd = sc.parallelize(nodes_list)
    # Convert to DataFrame
    nodes_df = pd.DataFrame(nodes_list, columns=["latitude", "longitude"])
    nodes_df["nodeID"] = nodes_df.index

    # Read edges (distance)
    edges_distance_rdd = spark.read.text(edges_distance_file).rdd
    edges_list = []
    for line in edges_distance_rdd.collect():
        parts = line.value.strip().split()
        if len(parts) == 3:
            try:
                source, dest, distance = int(parts[0]), int(parts[1]), float(parts[2])
                edges_list.append((source, dest, distance))
            except ValueError:
                print(f"Skipping invalid line: {line.value.strip()}")
    edges_rdd = sc.parallelize(edges_list)
    edges_distance_df = pd.DataFrame(edges_list, columns=["source", "destination", "distance"])

    # Read edges (travel time)
    edges_travel_rdd = spark.read.text(edges_travel_file).rdd
    edges_travel_list = []
    for line in edges_travel_rdd.collect():
        parts = line.value.strip().split()
        if len(parts) == 3:
            try:
                source, dest, travel_time = int(parts[0]), int(parts[1]), float(parts[2])
                edges_travel_list.append((source, dest, travel_time))
            except ValueError:
                print(f"Skipping invalid line: {line.value.strip()}")
    edges_travel_rdd = sc.parallelize(edges_travel_list)
    edges_travel_df = pd.DataFrame(edges_travel_list, columns=["source", "destination", "travel_time"])

    # Merge edge data
    edges_df = pd.merge(edges_distance_df, edges_travel_df, on=["source", "destination"], how="inner")
    print(edges_df.head())

    # Convert nodes to GeoDataFrame
    nodes_gdf = gpd.GeoDataFrame(
        nodes_df,
        geometry=gpd.points_from_xy(nodes_df.longitude, nodes_df.latitude),
        crs="EPSG:4326")

    # Create GeoDataFrame for edges
    def create_line_geometry(row):
        source_geom = nodes_gdf.loc[row["source"], "geometry"]
        dest_geom = nodes_gdf.loc[row["destination"], "geometry"]
        return LineString([source_geom, dest_geom])

    edges_gdf = gpd.GeoDataFrame(edges_df, geometry=edges_df.apply(create_line_geometry, axis=1), crs="EPSG:4326")

    # Export nodes to GeoJSON
    nodes_gdf.to_file(output_nodes_geojson, driver="GeoJSON")
    print(f"Node GeoJSON file saved as '{output_nodes_geojson}'")

    # Export edges to GeoJSON
    edges_gdf.to_file(output_edges_geojson, driver="GeoJSON")
    print(f"Edge GeoJSON file saved as '{output_edges_geojson}'")


def compute_accessibility_index_all_in_one(nodes_file, edges_file, output_file, 
                                           region_lat_min=22.27, region_lat_max=22.29, 
                                           region_lon_min=114.15, region_lon_max=114.18):
    """Compute the Accessibility Index for nodes in the popular region and export to GeoJSON."""
    
    # Initialize Spark Session
    spark = SparkSession.builder.appName("AccessibilityAnalysis").getOrCreate()
    sc = spark.sparkContext

    # Load Nodes Data
    nodes_rdd = spark.read.text(nodes_file).rdd
    nodes_rdd = nodes_rdd.zipWithIndex().filter(lambda x: x[1] > 0)  # Skip first line

    nodes_list = []
    for i, (line, _) in enumerate(nodes_rdd.collect()):
        parts = line.value.strip().split()
        if len(parts) == 2:
            try:
                lat, lon = float(parts[0]), float(parts[1])
                if region_lat_min <= lat <= region_lat_max and region_lon_min <= lon <= region_lon_max:
                    nodes_list.append((i, lat, lon))
            except ValueError:
                print(f"Skipping invalid line: {line.value.strip()}")

    # Create DataFrame for Nodes
    nodes_df = pd.DataFrame(nodes_list, columns=["nodeID", "latitude", "longitude"])

    # Load Edges Data
    edges_rdd = spark.read.text(edges_file).rdd
    edges_list = []

    for line in edges_rdd.collect():
        parts = line.value.strip().split()
        if len(parts) == 3:
            try:
                source, dest, travel_time = int(parts[0]), int(parts[1]), float(parts[2])
                if travel_time > 0:  # Avoid division by zero
                    edges_list.append((source, dest, travel_time))
            except ValueError:
                print(f"Skipping invalid line: {line.value.strip()}")

    # Create DataFrame for Edges
    edges_df = pd.DataFrame(edges_list, columns=["source", "destination", "travel_time"])

    # Build the NetworkX Graph
    G = nx.Graph()
    for _, row in edges_df.iterrows():
        G.add_edge(row["source"], row["destination"], weight=row["travel_time"])

    # Compute Accessibility Index for each node
    accessibility_scores = {}
    total_nodes = len(nodes_df)
    for idx, node in enumerate(nodes_df["nodeID"]):
        total_score = 0
        try:
            # Get shortest paths from the node to others using travel time as weight
            shortest_paths = nx.single_source_dijkstra_path_length(G, node, weight="weight")
            for _, travel_time in shortest_paths.items():
                total_score += 1 / travel_time if travel_time > 0 else 0  # Avoid division by zero
            accessibility_scores[node] = total_score
        except nx.NetworkXNoPath:
            accessibility_scores[node] = 0  # No connectivity

        # Print the number of processed nodes
        if (idx + 1) % 100 == 0:  # Print progress every 100 nodes
            print(f"Processed {idx + 1} out of {total_nodes} nodes.")

    # Map Accessibility Index scores to nodes DataFrame
    nodes_df["Accessibility Index"] = nodes_df["nodeID"].map(accessibility_scores)

    # Convert to GeoDataFrame
    nodes_gdf = gpd.GeoDataFrame(
        nodes_df,
        geometry=gpd.points_from_xy(nodes_df.longitude, nodes_df.latitude),
        crs="EPSG:4326"
    )

    # Export to GeoJSON
    nodes_gdf.to_file(output_file, driver="GeoJSON")
    print(f"Saved: {output_file}")


def compute_pagerank(nodes_file, edges_file, output_file, 
                     region_lat_min=22.27, region_lat_max=22.29, 
                     region_lon_min=114.15, region_lon_max=114.18, 
                     alpha=0.85, max_iter=100, tol=1e-6):
    """Compute PageRank for nodes within the specified region and export to GeoJSON."""

    # Initialize Spark Session
    spark = SparkSession.builder.appName("PageRank").getOrCreate()
    sc = spark.sparkContext

    # Load Nodes Data
    nodes_rdd = spark.read.text(nodes_file).rdd
    nodes_rdd = nodes_rdd.zipWithIndex().filter(lambda x: x[1] > 0)  # Skip first line

    nodes_list = []
    for i, (line, _) in enumerate(nodes_rdd.collect()):
        parts = line.value.strip().split()
        if len(parts) == 2:
            try:
                lat, lon = float(parts[0]), float(parts[1])
                if region_lat_min <= lat <= region_lat_max and region_lon_min <= lon <= region_lon_max:
                    nodes_list.append((i, lat, lon))
            except ValueError:
                print(f"Skipping invalid line: {line.value.strip()}")

    # Create DataFrame for Nodes
    nodes_df = pd.DataFrame(nodes_list, columns=["nodeID", "latitude", "longitude"])

    # Load Edges Data
    edges_rdd = spark.read.text(edges_file).rdd
    edges_list = []

    for line in edges_rdd.collect():
        parts = line.value.strip().split()
        if len(parts) == 3:
            try:
                source, dest, travel_time = int(parts[0]), int(parts[1]), float(parts[2])
                if travel_time > 0:  # Avoid division by zero
                    edges_list.append((source, dest, travel_time))
            except ValueError:
                print(f"Skipping invalid line: {line.value.strip()}")

    # Create DataFrame for Edges
    edges_df = pd.DataFrame(edges_list, columns=["source", "destination", "travel_time"])

    # Build the NetworkX Graph
    G = nx.DiGraph()  # Directed graph for PageRank computation
    for _, row in edges_df.iterrows():
        G.add_edge(row["source"], row["destination"], weight=row["travel_time"])

    # Initialize PageRank
    pagerank = {node: 1.0 / len(G.nodes) for node in G.nodes}
    
    # PageRank Iteration (Power iteration method)
    for iteration in range(max_iter):
        new_pagerank = {}
        for node in G.nodes:
            rank_sum = 0
            for neighbor in G.predecessors(node):
                rank_sum += pagerank[neighbor] / len(list(G.neighbors(neighbor)))  # Normalize by the number of neighbors
            new_pagerank[node] = (1 - alpha) / len(G.nodes) + alpha * rank_sum
        
        # Convergence check
        diff = sum(abs(new_pagerank[node] - pagerank[node]) for node in G.nodes)
        pagerank = new_pagerank

        # Print progress
        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Convergence Difference: {diff:.6f}")
        
        if diff < tol:
            print(f"Converged after {iteration} iterations.")
            break

    # Assign PageRank values to the nodes DataFrame
    nodes_df["PageRank"] = nodes_df["nodeID"].map(pagerank)

    # Create GeoDataFrame for Nodes
    nodes_gdf = gpd.GeoDataFrame(
        nodes_df,
        geometry=gpd.points_from_xy(nodes_df.longitude, nodes_df.latitude),
        crs="EPSG:4326"
    )

    # Export to GeoJSON
    nodes_gdf.to_file(output_file, driver="GeoJSON")
    print(f"PageRank values saved to: {output_file}")

# Example usage
compute_pagerank(
    "hdfs://spark-yarn-master:8080/data/Hongkong/Hongkong.co", 
    "hdfs://spark-yarn-master:8080/data/Hongkong/Hongkong.road-t", 
    "pagerank_values.geojson"
)



def compute_clustering(nodes_file, edges_file, output_file,
                       region_lat_min=22.27, region_lat_max=22.29, 
                       region_lon_min=114.15, region_lon_max=114.18):
    """Cluster nodes in the specified region based on modularity and export to GeoJSON."""

    # Initialize Spark Session
    spark = SparkSession.builder.appName("Clustering").getOrCreate()
    sc = spark.sparkContext

    # Load Nodes Data
    nodes_rdd = spark.read.text(nodes_file).rdd
    nodes_rdd = nodes_rdd.zipWithIndex().filter(lambda x: x[1] > 0)  # Skip first line

    nodes_list = []
    for i, (line, _) in enumerate(nodes_rdd.collect()):
        parts = line.value.strip().split()
        if len(parts) == 2:
            try:
                lat, lon = float(parts[0]), float(parts[1])
                if region_lat_min <= lat <= region_lat_max and region_lon_min <= lon <= region_lon_max:
                    nodes_list.append((i, lat, lon))
            except ValueError:
                print(f"Skipping invalid line: {line.value.strip()}")

    # Create DataFrame for Nodes
    nodes_df = pd.DataFrame(nodes_list, columns=["nodeID", "latitude", "longitude"])

    # Load Edges Data
    edges_rdd = spark.read.text(edges_file).rdd
    edges_list = []

    for line in edges_rdd.collect():
        parts = line.value.strip().split()
        if len(parts) == 3:
            try:
                source, dest, travel_time = int(parts[0]), int(parts[1]), float(parts[2])
                if travel_time > 0:  # Avoid division by zero
                    edges_list.append((source, dest, travel_time))
            except ValueError:
                print(f"Skipping invalid line: {line.value.strip()}")

    # Create DataFrame for Edges
    edges_df = pd.DataFrame(edges_list, columns=["source", "destination", "travel_time"])

    # Build the NetworkX Graph
    G = nx.Graph()  # Undirected graph for clustering
    for _, row in edges_df.iterrows():
        G.add_edge(row["source"], row["destination"], weight=row["travel_time"])

    # Apply Louvain method for community detection
    print("Starting community detection using Louvain method...")

    # Get the best partition using Louvain's algorithm
    partition = community.best_partition(G, weight='weight')

    # Create a dictionary to store nodeID and their corresponding community
    node_community = partition  # No need for manual mapping, since best_partition already provides this

    # Assign community labels to the nodes DataFrame
    nodes_df["Community"] = nodes_df["nodeID"].map(node_community)

    # Print progress of processed nodes
    for idx, row in nodes_df.iterrows():
        if idx % 100 == 0:  # Print progress every 100 nodes
            print(f"Processed {idx + 1} nodes.")

    # Create GeoDataFrame for Nodes with Community Labels
    nodes_gdf = gpd.GeoDataFrame(
        nodes_df,
        geometry=gpd.points_from_xy(nodes_df.longitude, nodes_df.latitude),
        crs="EPSG:4326"
    )

    # Export to GeoJSON
    nodes_gdf.to_file(output_file, driver="GeoJSON")
    print(f"Clustering results saved to: {output_file}")


# Example usage
compute_clustering(
    "hdfs://spark-yarn-master:8080/data/Hongkong/Hongkong.co", 
    "hdfs://spark-yarn-master:8080/data/Hongkong/Hongkong.road-t", 
    "clustering_results.geojson"
)



# File paths for output GeoJSON files
output_nodes_geojson = "nodes.geojson"
output_edges_geojson = "edges.geojson"




# Output file names
output_betweenness_geojson = "betweenness_centrality.geojson"
output_closeness_geojson = "closeness_centrality.geojson"
output_eigenvector_geojson = "eigenvector_centrality.geojson"
output_degree_geojson = "degree_centrality.geojson"


