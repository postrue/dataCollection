import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from scipy import stats


def logisticRegression(X_data, labels):
    X_train, X_test, y_train, y_test = train_test_split(X_data, labels, test_size=0.2, random_state=42)

    model = LogisticRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

# Function to remove features with p-value greater than 0.05
def remove_high_pvalue_features(data, group_labels, alpha=0.05):
    features_to_keep = []
    
    for feature_idx in range(data.shape[1]):  # Loop through each feature
        # Get the values of the feature for both groups
        group_0 = data[group_labels == 0][:, feature_idx]
        group_1 = data[group_labels == 1][:, feature_idx]
        
        # Perform a t-test (or another test)
        u_stat, p_value = mannwhitneyu(group_0, group_1)
        
        # Check if the p-value is below the threshold (0.05 by default)
        if p_value <= alpha:
            features_to_keep.append(feature_idx)  # Keep the feature if p-value is low enough

    # Filter data to keep only features that passed the p-value test
    filtered_data = data[:, features_to_keep]
    
    print(f"Features kept: {len(features_to_keep)} out of {data.shape[1]}: {features_to_keep}")
    
    return filtered_data, features_to_keep


# Function to remove outliers across all features using Z-score
def remove_outliers(data, threshold=3):
    z_scores = np.abs(stats.zscore(data))  # Compute Z-scores for each feature
    # Keep rows where Z-scores for all features are within the threshold
    return data[(z_scores < threshold).all(axis=1)]



# Function to analyze the difference in a specific feature between two groups
def analyze_feature(feature_index, group0, group1, true_labels):
    # Extract the values for the selected feature
    feature_values_0 = group0[:, feature_index]
    feature_values_1 = group1[:, feature_index]
    
    # Perform t-test or Mann-Whitney U test (if non-parametric test is preferred)
    u_stat, p_value = mannwhitneyu(feature_values_0, feature_values_1)

    # Print test results
    print(f"Feature {feature_index} - t-test p-value: {p_value:.5f}")
    
    # Visualization
    plt.figure(figsize=(10, 6))

    sns.boxplot(x=true_labels, y=np.hstack((feature_values_0, feature_values_1)), palette=['blue', 'red'])
    plt.title(f"Feature {feature_index} Comparison between Groups")
    plt.xlabel("Group")
    plt.ylabel(f"Feature {feature_index} Values")
    plt.xticks([0, 1], ['Group 0', 'Group 1'])
    plt.show()

# Function to visualize all features as subplots
def analyze_all_features(group0, group1, true_labels):
    fig, axes = plt.subplots(3, 4)  # 3x3 grid for 9 features
    axes = axes.ravel()  # Flatten the axes array for easier indexing

    for i in range(12):
        feature_values_0 = group0[:, i]
        feature_values_1 = group1[:, i]
        
        # Perform t-test
        u_stat, p_value = mannwhitneyu(feature_values_0, feature_values_1)
        print(f"Feature {i+1} - t-test p-value: {p_value:.5f}")
        
        # Combine data from both groups for visualization
        combined_feature_values = np.hstack((feature_values_0, feature_values_1))
        
        # Create boxplot for each feature
        sns.boxplot(ax=axes[i], x=true_labels, y=combined_feature_values, palette=['blue', 'red'])
        axes[i].set_title(f"Feature {i+1} (p-value: {p_value:.5f})")
        axes[i].set_xlabel("Group")
        axes[i].set_ylabel(f"Feature {i+1} Values")
        axes[i].set_xticks([0, 1], ['Group 0', 'Group 1'])
    
    plt.tight_layout()  # Adjust layout so subplots don't overlap
    plt.show()



# Assume you have two NumPy arrays: group0 and group1
group0 = np.load('relax_data.npy')  # Replace with actual group0 data
group1 = np.load('flex_data.npy')  # Replace with actual group1 data


# c0 = remove_outliers(group0)
# c1 = remove_outliers(group1)


c0 = remove_outliers(group0[:30,:])
c1 = remove_outliers(group0[30:,:])

c2 = remove_outliers(group1[:30,:])
c3 = remove_outliers(group1[30:,:])


# Stack the two arrays together
# X = np.vstack((c0,c1))
X = np.vstack((c0, c1, c2, c3))


# Create a label array for the true groups
# y_true = np.array([0] * c0.shape[0] + [1] * c1.shape[0]) 
y_true = np.array([0] * c0.shape[0] + [1] * c1.shape[0] + [2] * c2.shape[0] + [3] * c3.shape[0]) 





# Run the function to filter out features with p-value > 0.05
# filtered_data, kept_features = remove_high_pvalue_features(X, y_true)
filtered_data = X

# Apply normalization
data_min = np.min(filtered_data, axis=0)  # Minimum value for each feature (column-wise)
data_max = np.max(filtered_data, axis=0)  # Maximum value for each feature (column-wise)

# Normalize each feature (column-wise)
normalized_data = (filtered_data - data_min) / (data_max - data_min)



# Run K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
y_pred = kmeans.fit_predict(normalized_data)



# Reduce dimensions to 2D using PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(normalized_data)




# Plot predicted clusters
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_reduced[y_pred == 0, 0], X_reduced[y_pred == 0, 1], color='blue')
plt.scatter(X_reduced[y_pred == 1, 0], X_reduced[y_pred == 1, 1], color='red')
plt.scatter(X_reduced[y_pred == 2, 0], X_reduced[y_pred == 2, 1], color='green')
plt.scatter(X_reduced[y_pred == 3, 0], X_reduced[y_pred == 3, 1], color='purple')
plt.title('K-means Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid()


# Plot true labels
plt.subplot(1, 2, 2)
plt.scatter(X_reduced[y_true == 0, 0], X_reduced[y_true == 0, 1], color='blue', label='relaxed_0_0')
plt.scatter(X_reduced[y_true == 1, 0], X_reduced[y_true == 1, 1], color='red', label='relaxed_0_1')
plt.scatter(X_reduced[y_true == 2, 0], X_reduced[y_true == 2, 1], color='green', label='flexed_0')
plt.scatter(X_reduced[y_true == 3, 0], X_reduced[y_true == 3, 1], color='purple', label='flexed_1')
plt.title('True Labels')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

logisticRegression(normalized_data, y_true)




