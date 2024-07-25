import fun
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob


def main():
    dir = input("name of directory containing sessions: ")
    directory = f'./{dir}'
    axnames=['X','Y','Z']
    decays = [[],[],[]]
    errors = [[],[],[]]
    averages = [[],[],[]] # final dimension will be 3 x numfiles
    labels = []
    numfiles = 0
    files = sorted(glob.glob(os.path.join(directory, '*')))
    for file in files:
        if os.path.isdir(file):
            seshname = file.split('/')[-1]
            numfiles += 1
            sl, il = fun.seshinfo(f'{dir}/{seshname}')
            n = (sl // il) + 1
            amplitudes = fun.amps(f'{dir}/{seshname}')
            params, err = fun.expfit(amplitudes)
            curravg = np.zeros((3,n)) # final dimension will be 3 x n
            for j in range(n):
                for i in range(3):
                    curravg[i][j] = params[j,i,1]
                    decays[i].append(params[j,i,1])
                    errors[i].append(err[j,i,0])
                labels.append(f'{seshname}_{j}')
            for i in range(3):
                row_averages = np.mean(curravg, axis=1)
                averages[i].append(row_averages[i])

    for i in range(len(labels)):
        parts = labels[i].split('_')
        labels[i] = f'{parts[0]}_{parts[1]}'

    no_dupes = list(dict.fromkeys(labels))

    x_values = [i for i in range(1, numfiles + 1) for _ in range(n)]

    # [x-axis decay, z-axis decay, x-axis R^2, z-axis R^2]
    # modelFeatures = np.zeros((numfiles*n, 2))
    # modelLabels = np.zeros(numfiles*n)

    # for i in range(numfiles*n):
    #     modelFeatures[i] = [decays[0][i], errors[0][i]]
    #     parts = labels[i].split('_')
    #     if parts[0] == "relax":
    #         modelLabels[i] = 0
    #     else:
    #         modelLabels[i] = 1
    
    # np.save('model-features.npy', modelFeatures)
    # np.save('model-labels.npy', modelLabels)
        
    for ax in range(3):

        # Create scatter plot
        plt.figure(figsize=(12, 6))

        # Use a colormap to represent strength
        scatter = plt.scatter(x_values, decays[ax], c=errors[ax],  cmap='viridis', alpha=0.6, edgecolor='w', linewidth=0.5)


        # Add a color bar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Strength')


        plt.xticks(ticks=x_values, labels=labels)
        plt.gca().set_xticklabels(labels, rotation=45)

        # Add labels and title
        plt.ylabel('decay rate')
        plt.title(f'Decay Rates for {axnames[ax]}-Axis')


        # Show plot
        plt.show()


        # vector with nine values
        vector = np.array(averages[ax])

        # Compute the similarity matrix
        similarity_matrix = np.abs(vector[:, np.newaxis] - vector)

        # Create a heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(similarity_matrix, annot=True, cmap='viridis', xticklabels=no_dupes, yticklabels=no_dupes)
        plt.title(f'Similarity Matrix for {axnames[ax]}-Axis Decays')
        plt.xlabel('Value')
        plt.ylabel('Value')
        plt.show()

                    
if __name__ == '__main__':
    main()