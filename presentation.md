
# report 1

The topics of our group is "Anomalous Diffusion in Membranes". Although we have only been working
on this project for one and a half days, we have a much more understanding of what we are doing and are going
to do in comparison to Monday morning, when we first knew that we had received this task. This is greatly 
based on the generous help of our three lovely advisors Loris, Emiliano and Vania. So we very much thank them for their
patience and kindness.

The outline of our presentation is divided into three stages.
First we would like to present the problem we want to solve.
To achieve the goals we have planned several steps.
We would like to show first the steps that we have already taken
and finally the steps still need to be done.

At the first stage, we would like to introduce the background of this project.
In this slide, you could see the pictures showing the cell membranes. The upper picture
shows the plain phospholipid bilayer. The second are more close to the reality, which
included some extra proteins. These proteins
are prevalent in the cell membrane for the sake of highly selective transportation of materials
through the cell membranes.  These pictures might be deceptive, because it might make you think
that the molecules are fixed to their positions. But actually, all the molecules are moving constantly
what we call the diffusive behavirors. 
What we want to study, is to characteriye the diffusion of these lipid molecules. The reason we study is 
not only because it is an interesting topic in and of itself but also in many biological interactions, the diffusion of pariticles
are the rate-limiting step. So to have a deeper understanding of the diffusion of molecules may help us to learn
the kinetics of many biological and chemical processes.




## TO DO
slide 1 add background introduction


## Appendix

A Phospholipid Bilayer
The plasma membrane is composed mainly of phospholipids, which consist of fatty acids and alcohol. The phospholipids in the plasma membrane are arranged in two layers, called a phospholipid bilayer. As shown in the Figure below, each phospholipid molecule has a head and two tails. The head “loves” water (hydrophilic) and the tails “hate” water (hydrophobic). The water-hating tails are on the interior of the membrane, whereas the water-loving heads point outwards, toward either the cytoplasm or the fluid that surrounds the cell.

Molecules that are hydrophobic can easily

A phospholipid is a lipid made of glycerol, two fatty acid tails, and a phosphate-linked head group. Biological membranes usually involve two layers of phospholipids with their tails pointing inward, an arrangement called a phospholipid bilayer.

Cholesterol, another lipid composed of four fused carbon rings, is found alongside phospholipids in the core of the membrane.



# report 2

Since the displacement of each particle is treated as a stochastic process, it is more meaningful to describe the diffusion behavior by considering the statistics of this process. One often used criterium is the mean square displacement. Furthermore, the msd it self is also a stochastic process. The analysis of msd can provide the whole picture of the diffusion behaviror, as you will see later. So we want to mainly focus on the analysis of the MSD. But before we pick a statistics of MSD,  it is instructive to first study the distribution of the MSD before we deside which statistics we shall use.
This picture shows the histogram displaying the distribution of the MSD of the x and y component, because we are interested mainly in the planar diffusion.  For this particular example, the unit for the time is given in ps. And the whole timescale is 500ps .The total number of particles chosen are 150. As you can see, for the time scale smaller than 150 ps, the distribution is more or less symmetric. But for larger time scale, the distribution has a significant skewness. As the average number increases, the distribution tends to be more normal distributed for small time scale, whereas the skewness in large time scale does not change much.
Since in the analysis, we have only chosen maximal 20 percent of the whole time scale, so we can assume that the distribution is quite normal, in pariticular, the mean or the median can be considered as a useful factor describing the MSD. 
Next, we want to show you the result of the averaged msd data.


# final

We would like to introduce you now the simple model for the analysis of the diffusion.

The characteric three region of the diffusion behavirors can be described by the three equations. These equations can be derived by solving the generalized langevian equaions.

As explained briefly by Mathias before, the exponents of the time t, in these three regions are given by 2, alphaa and 1. For the ballistic region, the constant kb T/m is the usual mean square velocity in one dimension As in this region, the molecules are free to move, we treat the dynamics in the xy direction and z direction as approximately the same and apply the factor 3. 
For the subdiffusion region, which is a consequence of the correlation of the force exerted on the molecules, we expect the equation to be of the form given by the second equation. the key constant to determine is the potent alpha, as you will see later, the alpha usually ranges from 0.5 to 0.7 and is dependent upon the composition of the membrane. As time goes to infinity. the correlation of the forces can be treated to be negligible and we would expect to see the normal brownian motion. The D_infty is the diffusion coefficient. We would also like to obtain the value of the diffusion coefficient

For the simple model, we try to fit the MSD with respect to time with these three equations and do a fitting to retrieve the constants that we 
are interested in.

The first question is how can we divide the three regions. Since we already see that the potents of the t are different in the three regions, so a natural idea is to do a logarithm on both sides, and the potent becomes the slope of the curve. By observing the slope of the curve to we can divide it into three regions. 

This figure shows the plotted slope with log log scale and you could see that the slope goes from 2 to around 0.7 and then rise up to 1. Based on this general idea, we divide the curve and after that we do a linear regression in the log log scale in each sub region. The final result can be seen from this figure. With the slope shown in the left top corner. This figure also gives us a general idea
of the magnitude of the transistion time between the regions. for the ballistic to subdifusion, is around 1 ps, and the subudiffusion to brownian is around 1ns

As said before, we are interested in the evolution of alpha and diffusion coefficient with respect to the cholesterol concentration. For the presentation we have used the DOPC membrane data for the alpha, and POPC  for the diffusion coefficient. Generally, they both have a similar behaviror, but for completeness, we use two different data to show you the results. 
For the alpha value, You could see that as the cholesterol concentration increases, the alpha decreases, which might be explained by the fact that the existence of chlesterol slows down the diffusion process.

For the brownian region, we use the example of POPC. We observe that a similar tendency for the diffusion coeffient. as the concentration of cholesterol increases, the diffusion coefficient decreases, which means that the fluidity of the membrane lowers.

So this is the simple model we have tried to explain the diffusion behaviror based on the asymptotic analysis. But we want to derive a general model that describes the model in the whole time scale. This leads us to the next section.