import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import load
from numpy import save
import random



# Load the files and put them as dataframe
class Dataset:

    def __init__(self, lista, names):
        self.lista = lista
        self.names = names

    def load_data(self):
        '''Function used to load data that is already preprocessed
        Returns dat and dataset. Dat is the numeric data and dataset is composed of images'''

        dset = []
        dat = []
        for i in range(7):
            dset.append(load('data/npy/' + self.names[i] + '_image.npy'))
            dat.append(load('data/npy/' + self.names[i] + '_np2.npy'))
        print('data loaded')
        print(dat[0])
        plt.imshow(dset[0][1], interpolation='none', aspect='auto')
        fig_size = plt.rcParams["figure.figsize"]
        plt.show()
        return dset, dat

    def preprocess(self,calculus = False):
        '''Function that takes the row data and make calculus on it
        This function creates dat and dset.
        dset is a np array that contains the images(the hexagons) and dat is data with (events, rows, features)
        Returns dat and dset'''
        # Count the number of events
        for i in range(7):
            self.lista[i] = pd.DataFrame(data=self.lista[i])

        events=np.empty((7))
        for i in range (len(events)):
            events[i]=len(self.lista[i].groupby([0]).mean())
        events=events.astype(int)

        count=0
        print('NUMBER OF IMAGES')
        for j in self.names:
            print(j,': ',events[count])
            count=count+1

        if calculus:
            # Intensity max and min of all the particles
            Imaximo=1458.8
            Iminimo=0

            for i in range(7):
                self.lista[i].loc[(self.lista[i][3] <0),[3]]=0
                self.lista[i][3]= (self.lista[i][3] - Iminimo) / (Imaximo - Iminimo)


            # Divide number of events by 100 and convert it to int
            for i in range(7):
                self.lista[i][0] = (self.lista[i][0]/100).astype(int)

            # Reescale x,y
            # X,Y min of every df
            Xminimo=1823
            Yminimo=7370
            for i in range(7):
                self.lista[i][2]=self.lista[i][2]-Yminimo
                self.lista[i][1]=self.lista[i][1]-Xminimo

            #Reescale pixels and convert it to int
            for i in range(7):
                self.lista[i][1]=(round(self.lista[i][1]/333)).astype(int)
                self.lista[i][2]=(round(self.lista[i][2]/192)).astype(int)
            print(self.lista[0])

        #The size of the images is 54x92
        #Convert to numpy and save the file
        for i in range(7):
            self.lista[i]=self.lista[i].to_numpy()

        '''for i in range(7):
            print('data/npy/'+names[i]+'_n.npy')
            save('data/npy/'+names[i]+'_n.npy', lista[i])'''


        # Now, create a file with the position of the pixel and its intensity
        rows=1854
        dat=[]
        for i in range(7):
            dat.append(np.zeros((events[i],rows,3)))


        # The number of rows of each event (for every particle) is 1854
        rows=1854

        for i in range(7):
            df_n=self.lista[i][:,1:4]
            k=0
            for m in range (events[i]):
                for j in range (1854):
                    dat[i][m][j]=df_n[k]  # df[[1,2,3]]
                    k=k+1

        # Save to npy file
        '''print('saving npy image...')
        for i in range(7):
            print('data/npy/'+names[i]+'_np2.npy')
            save('data/npy/'+names[i]+'_np2.npy', dat[i])'''

        # Create a squared image to feed the nn. Array with: events, rows and features
        print('''generating images...''')
        dset=[]
        for i in range (7):
            dset.append(np.zeros((events[i],int(self.lista[i][:,1].max())+1,int(self.lista[i][:,2].max())+1)) )

        for i in range(7):
            print(self.names[i])
            for m in range (events[i]):
                for j in range (1854):
                    a=int(dat[i][m][j][0])
                    b=int(dat[i][m][j][1])
                    dset[i][m][a,b]=dat[i][m][j][2]

        # Save the squared image
        '''print('saving squared image...')
        for i in range(7):
            print('data/npy/'+names[i]+'_image.npy')
            save('data/npy/'+names[i]+'_image.npy', dset[i])'''
        return self.lista,dat,dset

    def save_images(self, dset):
        '''Function that takes dset as an argument and saves 10 images for each particle'''

        # Save the images
        for k in range(7):
            for i in range(10):  # (len(dset))
                ax = plt.imshow(dset[k][i], cmap=plt.cm.binary, vmin=0, vmax=1)
                plt.savefig('images/hexagons/' + self.names[k] + '{0}.png'.format(i))
            if k==1:
                plt.imshow(dset[k][1], interpolation='none', aspect='auto')  # if vmin and vmax is removed, images with low intensities can be properly seen
                fig_size = plt.rcParams["figure.figsize"]
                plt.show()
            plt.close()
        return  None


    def shift(self, particle):
        '''Function that for a given particle, it shifts the images randomly. In this case it's only needed for gamma,
        because all the non-zero pixels are centered in the image
        Saves the .npy file with the images, and 100 plots as samples'''

        df1 = load('data/npy/'+particle+'_n.npy')
        df1 = pd.DataFrame(data=df1)
        df = load('data/npy/'+particle+'_np2.npy')

        # A dataset with the background noise is created
        conjunto = df1.loc[df1[4] == 0][3].values.copy()

        random.choice(conjunto)

        # create a squared image to feed to the neural network
        dseta = np.zeros((11999, int(df[0][:, 0].max()) + 1,
                          int(df[0][:, 1].max()) + 1))  # for the original image (to compare original and shifted images)
        dsetb = np.zeros((11999, int(df[0][:, 0].max()) + 1, int(df[0][:, 1].max()) + 1))  # for the shifted image

        for m in range(11999):  # for each event
            print('evento: ', m)
            random1 = int(random.uniform(0, 2))  # to choose left or right
            random2 = int(random.uniform(0, 2))  # to choose up or down

            if (random1 == 0):  # SHIFT IMAGE ON THE LEFT

                if (random2 == 0):  # SHIFT IMAGE ON THE LEFT UP
                    p = int(random.uniform(0, 30))  # how many pixels to move it
                    for j in range(1854):
                        a = int(df[m][j][0])  # take the x position from the original image
                        b = int(df[m][j][1])  # take the y position from the original image
                        dseta[m][a, b] = df[m][j][2]  # assign it to the image not moved

                        # if the image is moved 3 pixels on the left, the three left most pixels colums are filled with noise.
                        # check whether the pixel(a,b) belongs to the pixels to move or to fill with background noise
                        if (b < 55 - p):  # pixel(a,b) belongs to the pixels to move
                            k = 0
                            try:
                                c = np.where((df[m][:, 0] == a + p) & (df[m][:, 1] == b + p))[0][
                                    0]  # take the point from the especified position in the original image
                                k = 1
                            except:
                                pass

                            # try/except in case a point is not found (for example, it is out of the hexagon) and therefore the program raises error, and if it happens, a point from noise data is assigned
                            if (k == 1):
                                dsetb[m][a, b] = df[m][c][2]
                            else:
                                dsetb[m][a, b] = random.choice(conjunto)

                        else:  # pixel(a,b) belongs to the pixels to fill with background noise
                            dsetb[m][a, b] = random.choice(conjunto)

                else:  # SHIFT IMAGE ON THE LEFT DOWN (same structure as above)
                    p = int(random.uniform(0, 8))
                    for j in range(1854):
                        a = int(df[m][j][0])
                        b = int(df[m][j][1])
                        dseta[m][a, b] = df[m][j][2]

                        if (b < 93 - p):
                            k = 0
                            try:
                                c = np.where((df[m][:, 0] == a - p) & (df[m][:, 1] == b + p))[0][
                                    0]  # take the point from the especified position in the original image
                                k = 1
                            except:
                                pass

                            if (k == 1):
                                dsetb[m][a, b] = df[m][c][2]
                            else:
                                dsetb[m][a, b] = random.choice(conjunto)

                        else:
                            dsetb[m][a, b] = random.choice(conjunto)



            else:  # SHIFT IMAGE ON THE RIGHT DOWN

                if (random2 == 0):  # SHIFT IMAGE ON THE RIGHT DOWN
                    p = int(random.uniform(0, 10))
                    for j in range(1854):
                        a = int(df[m][j][0])
                        b = int(df[m][j][1])
                        dseta[m][a, b] = df[m][j][2]

                        if (b > p):
                            k = 0
                            try:
                                c = np.where((df[m][:, 0] == a - p) & (df[m][:, 1] == b - p))[0][
                                    0]  # take the point from the especified position in the original image
                                k = 1
                            except:
                                pass

                            if (k == 1):
                                dsetb[m][a, b] = df[m][c][2]
                            else:
                                dsetb[m][a, b] = random.choice(conjunto)

                        else:
                            dsetb[m][a, b] = random.choice(conjunto)

                else:  # SHIFT IMAGE ON THE RIGHT UP
                    p = int(random.uniform(0, 30))
                    for j in range(1854):
                        a = int(df[m][j][0])
                        b = int(df[m][j][1])
                        dseta[m][a, b] = df[m][j][2]

                        if (b > p):
                            k = 0
                            try:
                                c = np.where((df[m][:, 0] == a + p) & (df[m][:, 1] == b - p))[0][
                                    0]  # take the point from the especified position in the original image
                                k = 1
                            except:
                                pass

                            if (k == 1):
                                dsetb[m][a, b] = df[m][c][2]
                            else:
                                dsetb[m][a, b] = random.choice(conjunto)

                        else:
                            dsetb[m][a, b] = random.choice(conjunto)

        save('data/npy/gamma_image_moved.npy', dsetb)  # save the images as a npy file
        # Save the images
        print('saving shifted images...')
        for i in range(100):  # (len(dset))
            plt.imshow(dsetb[i], cmap=plt.cm.binary, vmin=0, vmax=1)
            plt.savefig("images/gamma_moved/gamma{0}.png".format(i))

        return None








