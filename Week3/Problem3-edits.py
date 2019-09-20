import numpy as np
import sympy as sp
from scipy import stats as spstats
import matplotlib.pyplot as plt

# so we need to find a way to compose a mixture of Pr1(x) and Pr2(x)

class GaussianDistributionPlotter:
    def __init__(self):
        self.i = 0
        self.mu = np.array([0, 1])
        self.sigma = np.array([(5, -2), (-2, 7)])
        self.muX = np.array([0, 0])
        self.sigmaX = np.array([(5, 0), (0, 0)])
        
        self.muXGivenYIs1 = self.mu[0] + self.sigma[0][1]*(1 / self.sigma[1][1])*(1 - self.mu[1])
        self.muXGivenYIs1 = np.array([self.muXGivenYIs1, 0])
        
        self.sigmaXGivenYIs1 = self.sigma[0][0] - self.sigma[0][1]*(1 / self.sigma[1][1])*self.sigma[1][0]
        self.sigmaXGivenYIs1 = np.array([(self.sigmaXGivenYIs1, 0),(0, 0)])
        
         # print("mu: {}\nsigma:\n{}".format(mu, sigma))
         # print("muX:\n{}".format(muX))
         # print("sigmaX:\n{}".format(sigmaX))
         # print("muXGivenYIs1:\n{}".format(muXGivenYIs1))
         # print("sigmaXGivenYIs1:\n{}".format(sigmaXGivenYIs1))
        
    def __plot(self, plot=None, data=None):
        plotChoices = ["XY", "X", "X|Y=1"]
        
        if data is None or plot is None or plot not in plotChoices:
            raise ValueError()
             
        plt.figure(self.i)
        self.i += 1
            
        if plot == "XY":
            samplesXY = data[0]
            X, Y, Z = data[1]
            plt.hist2d(samplesXY.T[0], samplesXY.T[1], bins=150)
            plt.contour(X, Y, Z, linewidths=2.5)
            plt.colorbar()
            plt.xlabel('$y_1$')
            plt.ylabel('$y_2$')
            plt.title("Density of $Pr(x,y)$ calculated by KDE (contour) vs analysis (histogram)")
            
        elif plot == "X":             
            kdePDFOfX = data[0]
            analysisPDFOfX = data[1]
            Xpoints = data[2]
             
            plt.plot(Xpoints, kdePDFOfX, '-o', label="KDE")
            plt.plot(Xpoints, analysisPDFOfX, '--', label="Analysis")
            plt.xlabel('$x$')
            plt.ylabel('$Pr(x)$')
            plt.title("Density of $Pr(x)$ calculated by analysis vs. by KDE")
            plt.legend()
             
        elif plot == "X|Y=1":             
            kdePDFOfXGivenYIs1 = data[0]
            analysisPDFOfXGivenYIs1 = data[1]
            XGivenYIs1points = data[2]
             
            plt.plot(XGivenYIs1points, kdePDFOfXGivenYIs1, '-o', label="KDE")
            plt.plot(XGivenYIs1points, analysisPDFOfXGivenYIs1, '--', label="Analysis")
            plt.xlabel('$x$')
            plt.ylabel('$Pr(x \mid y=1)$')
            plt.plot()
            plt.title("Density of $Pr(x \mid y=1)$ calculated by analysis vs. by KDE")
            plt.legend()
        
    def computeAndPlotSamples(self):
         # returns a (1000, n) array of samples --- 1000 samples for each distribution, n Gaussian functions
         samplesXY = np.random.multivariate_normal(self.mu, self.sigma, 1000)
         samplesX = np.random.multivariate_normal(self.muX, self.sigmaX, 1000)
         samplesXGivenYIs1 = np.random.multivariate_normal(self.muXGivenYIs1, self.sigmaXGivenYIs1, 1000)
         # print("samplesXY size: {}\nsamplesX size: {}\nsamplesXGivenYIs1 size: {}".format(samplesXY.shape,
         #                                                                                  samplesX.shape,
         #                                                                                  samplesXGivenYIs1.shape))

         XYpoints = np.zeros((2, 125))
         XYpoints[0] = np.linspace(-15, 15, 125)
         XYpoints[1] = np.linspace(-15, 15, 125)
         Xpoints = np.linspace(-25, 25, 125)
         XGivenYIs1points = np.linspace(-25, 25, 125)

         tempnorm = spstats.norm(self.muX[0] , np.sqrt(self.sigmaX[0][0]))
         analysisPDFOfX = np.array([tempnorm.pdf(x) for x in Xpoints])

         tempnorm = spstats.norm(self.muXGivenYIs1[0], np.sqrt(self.sigmaXGivenYIs1[0][0]))
         analysisPDFOfXGivenYIs1 = np.array([tempnorm.pdf(x) for x in XGivenYIs1points])

         # computing and evaluate the kernel density estimates
         kdeXY = spstats.gaussian_kde(samplesXY.T)
         kdeX = spstats.gaussian_kde(samplesX[:, 0].T)
         kdePDFOfX = kdeX.evaluate(points=Xpoints)
         kdeXGivenYIs1 = spstats.gaussian_kde(samplesXGivenYIs1[:, 0].T)
         kdePDFOfXGivenYIs1 = kdeXGivenYIs1.evaluate(points=XGivenYIs1points)
        
         X, Y = np.meshgrid(XYpoints[0], XYpoints[1])
         positions = np.vstack([Y.ravel(), X.ravel()])
         Z = np.reshape(kdeXY(positions).T, X.shape)
                
         self.__plot(plot="XY", data=(samplesXY, (X, Y, Z)))
         self.__plot(plot="X", data=(kdePDFOfX, analysisPDFOfX, Xpoints))
         self.__plot(plot="X|Y=1", data=(kdePDFOfXGivenYIs1, 
                                       analysisPDFOfXGivenYIs1, 
                                       XGivenYIs1points))
         
if __name__ == "__main__":
    gaussDP = GaussianDistributionPlotter()
    gaussDP.computeAndPlotSamples()