# Money Won't Buy You Safety

## Are you better off in a less developed country?

I recently came across a graph that shocked me as it seemed to imply the above. The plot in question is the one shown below. It shows total cases/deaths per million and the human development index (HDI) of the countries. It has a logarithmic y axis and a linear x axis and shows that as HDI increases, the cases and deaths per million also increase. To me, this is a crazy claim as I would initially think that more developed countries would have few cases and deaths due to better healthcare infrastructure and access to tests and vaccines.


```python
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime as dt
import scipy.stats as ss
%matplotlib inline
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_columns', None)
```


```python
df = pd.read_csv("owid-covid-data.csv")
```


```python
df2 = df[["total_tests","iso_code","location","date","total_cases_per_million","total_cases",
          "total_deaths","total_deaths_per_million","human_development_index","population"]]
dfHDI = df2.loc[df['date'] == "2021-03-23"]
dfHDI = dfHDI.drop([798,4023,23782,24208,50944,52164,64162,75743])
dfHDI = dfHDI.reset_index()
x = dfHDI['human_development_index']
y1 = dfHDI['total_cases_per_million']
y2 = dfHDI['total_deaths_per_million']
fig1 = plt.figure(figsize = (10, 10))
ax1 = fig1.add_subplot()
plt.scatter(x, y1, s = 3, alpha=1, c="b")
plt.scatter(x, y2, s = 3, alpha=1, c="r")
ax1.set_yscale('log')
plt.yticks(fontsize = 10)
plt.xticks(fontsize = 10)
plt.ylabel("Total Cases/Deaths Per Million")
plt.xlabel("HDI")
plt.title("Total Cases/Deaths given HDI", fontsize = 12)
plt.legend((("Cases"), ("Deaths")), loc = "upper left", fontsize = 10)
plt.show();
```


![png](output_5_0.png)


The graph does show an upwards trend for both <font color=blue>cases</font> and <font color=red>deaths</font>, although not a very tight correlation, with a few anomalous results that show the same levels of cases and deaths at higher HDIs than at lower ones. The correlation coefficients for cases and deaths are <font color=blue>0.74953</font> and <font color=red>0.72743</font> respectively which, in my opinion, are high enough to discount the anomalies and investigate why more developed countries were seemingly hit harder by the COVID-19 pandemic.
To explore the reasons behind this plot, I am going to look into the following:
- The testing data
- The obesity rate in the countries
- The mortality rate of COVID-19 in each country

#### Calculating logs


```python
print("Natural log of cases: ",np.log(dfHDI['total_cases_per_million']).corr(dfHDI['total_cases_per_million']))
print("Natural log of deaths:",np.log(dfHDI['total_deaths_per_million']).corr(dfHDI['total_cases_per_million']))
```

    Natural log of cases:  0.7495267974307972
    Natural log of deaths: 0.7274268041525757
    

### Exploring the COVID-19 Testing Data

I think the higher number of cases can be attributed to the fact that more developed countries had access to more tests, and could better implement a widespread testing procedure, and therefore would get more official positive test results on record. To explore this, I used the ‘total tests’ column from a week before the data from the initial plot (in order to make my sample size as large as possible due to delayed reporting) and then divided this by the nation’s population. This gave me a rough estimate of the proportion of the population that had taken a test (rough because some people would have taken many tests whilst some will have taken none) and then I used this to compare testing between nations. 


```python
dfTEST = df.loc[df['date'] == "2021-03-16"]
dfTEST = dfTEST[["total_tests","total_tests_per_thousand","new_tests","location","date","human_development_index","population"]]
dfTEST['pop% Test'] = dfTEST['total_tests']*100/dfTEST['population']
dfTEST = dfTEST.dropna()
```


```python
x = dfTEST['human_development_index']
y = dfTEST['pop% Test']
fig4 = plt.figure(figsize = (10, 10))
ax4 = fig4.add_subplot()
plt.scatter(x, y, s = 10, alpha = 1, c = "purple", label = "Tests")
plt.yticks(fontsize = 10)
plt.xticks(fontsize = 10)
plt.ylabel("Total Tests Over Population")
plt.xlabel("HDI")
plt.title("Total Tests given HDI", fontsize = 12)
plt.show();
```


![png](output_12_0.png)


The graph shows that as the HDI for a country increases, so does their percentage of tests over the population. Some nations were able to record enough tests for four times their population whilst others couldn’t hit 10%. Whilst this is not necessarily a causal relationship, I think it’s fair to say that HDI is a good indicator of whether or not a country conducted a lot of tests. This is because we had a large range of HDI’s (0.45 – 0.95) and only after 0.8 do we see a rise in the number of tests. To me this suggests that only nations who are highly developed had a large testing capacity. This could explain why we saw the upwards trend in the original graph as the more tests a country does, the more positive results they will get. Therefore developed countries who test a lot will have more positive results than less developed ones.

### Death Statistics

In order to understand the deaths trend shown in the original graph, I created some new statistics from the data so that I could grasp the numbers better. I calculated the percentage of cases that resulted in a death, along with the percentage of the population that had a positive case or a confirmed death. I will use these to produce more plots and to try and explain the correlations in the first plot. 


```python
dfHDI['Perc_Died'] = dfHDI['total_deaths']*100/dfHDI['population']
dfHDI['Fatal_Cases'] = dfHDI['total_deaths']*100/dfHDI['total_cases']
dfHDI['Perc_Infected'] = dfHDI['total_cases']*100/dfHDI['population']
dfHDI[['location','Fatal_Cases','Perc_Died','Perc_Infected']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>location</th>
      <th>Fatal_Cases</th>
      <th>Perc_Died</th>
      <th>Perc_Infected</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>4.389697</td>
      <td>0.006335</td>
      <td>0.144309</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albania</td>
      <td>1.769432</td>
      <td>0.074918</td>
      <td>4.234033</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Algeria</td>
      <td>2.635175</td>
      <td>0.006992</td>
      <td>0.265328</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Andorra</td>
      <td>0.974894</td>
      <td>0.146250</td>
      <td>15.001618</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Angola</td>
      <td>2.434096</td>
      <td>0.001613</td>
      <td>0.066250</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>189</th>
      <td>Venezuela</td>
      <td>0.990768</td>
      <td>0.005314</td>
      <td>0.536321</td>
    </tr>
    <tr>
      <th>190</th>
      <td>Vietnam</td>
      <td>1.359223</td>
      <td>0.000036</td>
      <td>0.002645</td>
    </tr>
    <tr>
      <th>191</th>
      <td>Yemen</td>
      <td>21.733112</td>
      <td>0.002632</td>
      <td>0.012110</td>
    </tr>
    <tr>
      <th>192</th>
      <td>Zambia</td>
      <td>1.365538</td>
      <td>0.006446</td>
      <td>0.472037</td>
    </tr>
    <tr>
      <th>193</th>
      <td>Zimbabwe</td>
      <td>4.128878</td>
      <td>0.010200</td>
      <td>0.247037</td>
    </tr>
  </tbody>
</table>
<p>194 rows × 4 columns</p>
</div>



Now that I have these statistics, I can use them to explore the deaths in a more indepth manor and hopefully come to a conclusion as to why deaths per million were higher for more developed nations. 

### Body Mass Index (BMI)


The idea of looking into BMI data stemmed from the claims that obese people were more at risk of death from COVID-19. The [Gov Website]( https://www.gov.uk/government/news/excess-weight-can-increase-risk-of-serious-illness-and-death-from-covid-19) makes this claim along with other sources and so I collected data from [NCD RisC]( https://ncdrisc.org/data-downloads-adiposity.html). I then processed it so that I could merge it with my existing data and produce a graph to show if deaths were more common in obese nations and whether or not these nations were developed or not. Unfortunately, the data set had different values for men and women and so I had to choose which one to use: ultimately going with the male data as more men have died from COVID-19 than women [(as seen in this article)]( https://www.nature.com/articles/s41467-020-19741-6).  


```python
BMI = pd.read_csv("NCD_RisC_Lancet_2017_BMI_age_standardised_country.csv", encoding= 'unicode_escape')
BMI = BMI.loc[BMI['Year'] == 2016]
BMIMen = BMI.loc[BMI['Sex'] == 'Men']
BMIMen = BMIMen.reset_index()
BMIMen = BMIMen[["ISO", "Mean BMI", "Prevalence of BMI>=30 kg/m² (obesity)", "Prevalence of BMI>=35 kg/m² (severe obesity)"]]
```


```python
mergedDF = dfHDI
mergedDF = mergedDF.merge(BMIMen, left_on = "iso_code", right_on = 'ISO')
```


```python
x = mergedDF['Prevalence of BMI>=30 kg/m² (obesity)']
y = mergedDF['Perc_Died']
fig3 = plt.figure(figsize = (10, 10))
ax3 = fig3.add_subplot()
plt.scatter(x, y, s = 5, alpha = 1, c = mergedDF.human_development_index, cmap = 'RdYlGn')
cbar = plt.colorbar()
cbar.ax.set_ylabel('HDI', rotation = 270)
plt.yticks(fontsize = 10)
plt.xticks(fontsize = 10)
plt.ylabel("% of Population That Died of COVID-19")
plt.xlabel("Prevalence of BMI>=30 kg/m² (obesity)")
plt.title("Total Cases/Deaths given HDI", fontsize = 12)
plt.show();
```


![png](output_22_0.png)


The graph does clearly show that as the prevalence of obesity rises, so does the percentage of the population that died from COVID-19. The plot also shows that as the HDI increases the prevalence of obesity does too. This is the link we need to account for the trend, seen in the original graph, that more developed countries have higher deaths per million than less developed ones. As their HDI rises, they are able to afford more food, leading to a more obese population which in turn means a higher mortality rate from COVID-19.

### Final Thoughts


```python
x = mergedDF['human_development_index']
y2 = mergedDF['Fatal_Cases']
y1 = mergedDF['Perc_Infected']
fig5 = plt.figure(figsize = (10, 10))
ax5 = fig5.add_subplot()
plt.scatter(x, y2, s = 5, alpha = 1, c = mergedDF.Perc_Infected, cmap = 'brg')
cbar = plt.colorbar()
cbar.set_label('Percentage of The Population That Has Been Infected', rotation = 270, labelpad = 15)
plt.yticks(fontsize = 10)
plt.xticks(fontsize = 10)
plt.ylabel("% of Fatal Cases")
plt.xlabel("HDI")
plt.title("Fatal Cases given HDI", fontsize = 12)
plt.show();
```


![png](output_25_0.png)


I’ve made this graph to show how these new statistics can change the perspective on the deaths amongst countries. Rather than having deaths per million on the y axis, I have the percentage of cases that were fatal. This should eliminate the problem of more developed countries being able to test more because it isn’t a measure of how many deaths, rather a proportion of fatal cases. It shows that for all HDI levels, the proportion of fatal cases remains relatively constant. This is more inline with what I expected when making the first plot because I didn’t think more developed countries would have more deaths. The deaths due to obesity could be counteracted by the better healthcare provision in more developed countries which is why we see no deviation from the trend at higher levels of HDI.

The colour scale shows the percentage of the population that has been infected and this matches the fatal cases almost perfectly which is expected. I did a colour scale rather than plotting the points because I think that it made it easier to read as there were fewer overlapping points. The green points show countries that had a high percentage of their population infected and they are towards the developed end of HDI but are much higher than those around them.     


```python
print("Correlation between % population that got infected and HDI: ",np.log(dfHDI['Perc_Infected']).corr(dfHDI['human_development_index']))
print("Correlation between % of fatal cases and HDI: ",np.log(dfHDI['Fatal_Cases']).corr(dfHDI['human_development_index']))
```

    Correlation between % population that got infected and HDI:  0.6396382538785145
    Correlation between % of fatal cases and HDI:  -0.12027319456756001
    

Having calculated the correlations, it's easier to see that there is not a strong correlation between fatal cases and HDI. This is further evidence for me that the initial graph was effected by how well a country was able to test it's cases and report it's deaths because as soon as you turn the data into proportions, the trends go away. We see no correlation between fatal cases and HDI but there is one between the reported percentage of the population that got infected and HDI which to me, confirms the fact that more developed countries were better able to report their positive cases and deaths but weren't any worse at handling cases as their proportion of fatal cases were the same as the less developed ones.

#### Works Cited
- NCD RisC. (2019). NATIONAL ADULT BODY-MASS INDEX. Retrieved April 25, 2021, from NCD RisC: https://ncdrisc.org/data-downloads-adiposity.html
- Peckham, H., de Gruijter, N., Raine, C., & al, e. (2020, December 9). Male sex identified by global COVID-19 meta-analysis as a risk factor for death and ITU admission. doi:https://doi.org/10.1038/s41467-020-19741-6
- Public Health England. (2020, July 2020). Excess weight can increase risk of serious illness and death from COVID-19. Retrieved April 25, 2021, from GOV.UK: https://www.gov.uk/government/news/excess-weight-can-increase-risk-of-serious-illness-and-death-from-covid-19




```python

```
