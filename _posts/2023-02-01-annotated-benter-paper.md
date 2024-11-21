---
layout: post
title: "Revisiting the Algorithm that Changed Horse Race Betting"
description: Explore Bill Benter’s iconic horse betting strategy, enhanced with modern code and contrasted against three decades of evolving data.
authors: [tlyleung, plyleung]
x: 78
y: 24
---

> The remarkable story of Bill Benter and how he amassed a staggering $1B fortune betting on horses in Hong Kong has been extensively documented in the article, [The Gambler Who Cracked the Horse-Racing Code](https://www.bloomberg.com/news/features/2018-05-03/the-gambler-who-cracked-the-horse-racing-code)[^chellel18]. In 1994, Benter published an academic paper titled [Computer Based Horse Race Handicapping and Wagering Systems: A Report](https://www.gwern.net/docs/statistics/decision/1994-benter.pdf)[^benter08]. In it, he documents the implementation of a successful horse race betting model, which by virtue of it being published, likely meant that the model was outdated and superceded by more sophisticated models. Despite this, the paper remains an insightful read, offering a rare glimpse into the lucrative application of mathematics to an unconventional field, made even more interesting by the substantial hardware and software limitations of the time.
> 
> In this post, we present an annotated version of the paper with added code blocks and blockquoted comments. The main text is all from the paper itself. Instead of building the fundamental and combined models, we'll be highlighting interesting aspects of the paper using the public estimate, derived from the [Hong Kong Jockey Club's historical win odds](https://racing.hkjc.com/racing/information/English/racing/LocalResults.aspx). We'll look at how the model calibration tables were generated in the original paper, assess how the public estimate has improved through the years and try our hand at fitting the adjustments factors from scratch using PyTorch. While the original paper uses data samples between 1986--1993, we'll also be using data from the subsequent three decades (1996--2003, 2006--2013 and 2016--2023) for comparison.


```python
import pandas as pd

DATA_FILE = "../data/HKJC/df_hkjc.csv"

DATE_RANGES = [
    ("1986-08-01", "1993-08-01"),
    ("1996-08-01", "2003-08-01"),
    ("2006-08-01", "2013-08-01"),
    ("2016-08-01", "2023-08-01"),
]

pd.read_csv(DATA_FILE, index_col=[0, 1, 2, 3])
```

<figure class="tabular-nums" markdown="1">
<figcaption>Table A: Historical Win Odds from Hong Kong Jockey Club</figcaption>

| date           | venue            | number | horse_name            | place | win_odds |
|----------------|------------------|--------|-----------------------|-------|---------:|
| **1979-09-22** | **Happy Valley** | **1**  | **Victorious II**     | 1     |      3.4 |
| **1979-09-22** | **Happy Valley** | **1**  | **Money First**       | 2     |     16.0 |
| **1979-09-22** | **Happy Valley** | **1**  | **Star Trouper**      | 3     |      3.2 |
| **1979-09-22** | **Happy Valley** | **1**  | **Red Rocker**        | 4     |     27.0 |
| **1979-09-22** | **Happy Valley** | **1**  | **New Gem**           | 5     |     27.0 |
| **…**          | **…**            | **…**  | **…**                 | …     |        … |
| **2023-07-16** | **Sha Tin**      | **9**  | **Master Of Fortune** | 5     |      5.1 |
| **2023-07-16** | **Sha Tin**      | **9**  | **Pegasus General**   | 6     |    122.0 |
| **2023-07-16** | **Sha Tin**      | **9**  | **Charity Bingo**     | 7     |     13.0 |
| **2023-07-16** | **Sha Tin**      | **9**  | **Chiu Chow Spirit**  | 8     |     10.0 |
| **2023-07-16** | **Sha Tin**      | **9**  | **So We Joy**         | 9     |     33.0 |

<p class="text-sm text-zinc-500 dark:text-zinc-400">337003 rows × 2 columns</p>
</figure>

## Abstract

This paper examines the elements necessary for a practical and successful computerized horse race handicapping and wagering system. Data requirements, handicapping model development, wagering strategy, and feasibility are addressed. A logit-based technique and a corresponding heuristic measure of improvement are described for combining a fundamental handicapping model with the public's implied probability estimates. The author reports significant positive results in five years of actual implementation of such a system. This result can be interpreted as evidence of inefficiency in pari-mutuel racetrack wagering. This paper aims to emphasize those aspects of computer handicapping which the author has found most important in practical application of such a system.

## Introduction

The question of whether a fully mechanical system can ever "beat the races" has been widely discussed in both the academic and popular literature. Certain authors have convincingly demonstrated that profitable wagering systems do exist for the races. The most well documented of these have generally been of the *technical* variety, that is, they are concerned mainly with the public odds, and do not attempt to predict horse performance from fundamental factors. Technical systems for place and show betting, (Ziemba and Hausch, 1987) and exotic pool betting, (Ziemba and Hausch, 1986) as well as the "odds movement" system developed by Asch and Quandt (1986), fall into this category. A benefit of these systems is that they require relatively little preparatory effort, and can be effectively employed by the occasional racegoer. Their downside is that betting opportunities tend to occur infrequently and the maximum expected profit achievable is usually relatively modest. It is debatable whether any racetracks exist where these systems could be profitable enough to sustain a full-time professional effort.

To be truly viable, a system must provide a large number of high advantage betting opportunities in order that a sufficient amount of expected profit can be generated. An approach which does promise to provide a large number of betting opportunities is to *fundamentally* handicap each race, that is, to empirically assess each horse's chance of winning, and utilize that assessment to find profitable wagering opportunities. A natural way to attempt to do this is to develop a computer model to estimate each horse's probability of winning and calculate the appropriate amount to wager.

A complete survey of this subject is beyond the scope of this paper. The general requirements for a computer based fundamental handicapping model have been well presented by Bolton and Chapman (1986) and Brecher (1980). These two references are "required reading" for anyone interested in developing such a system. Much of what is said here has already been explained in those two works, as is much of the theoretical background which has been omitted here. What the author would hope to add, is a discussion of a few points which have not been addressed in the literature, some practical recommendations, and a report that a *fundamental* approach can in fact work in practice.

## Features of the Computer Handicapping Approach

Several features of the computer approach give it advantages over traditional handicapping. First, because of its empirical nature, one need not possess specific handicapping expertise to undertake this enterprise, as everything one needs to know can be learned from the data. Second is the testability of a computer system. By carefully partitioning data, one can develop a model and test it on *unseen* races. With this procedure one avoids the danger of overfitting past data. Using this "holdout" technique, one can obtain a reasonable estimate of the system's real-time performance before wagering any actual money. A third positive attribute of a computerized handicapping system is its consistency. Handicapping races manually is an extremely taxing undertaking. A computer will effortlessly handicap races with the same level of care day after day, regardless of the mental state of the operator. This is a non-trivial advantage considering that a professional level betting operation may want to bet several races a day for extended periods.

The downside of the computer approach is the level of preparatory effort necessary to develop a winning system. Large amounts of past data must be collected, verified and computerized. In the past, this has meant considerable manual entry of printed data. This situation may be changing as optical scanners can speed data entry, and as more online horseracing database services become available. Additionally, several man-years of programming and data analysis will probably be necessary to develop a sufficiently profitable system. Given these considerations, it is clear that the computer approach is not suitable for the casual racegoer.

## Handicapping Model Development

The most difficult and time-consuming step in creating a computer based betting system is the development of the fundamental handicapping model. That is, the model whose final output is an estimate of each horse's probability of winning. The type of model used by the author is the multinomial logit model proposed by Bolton and Chapman (1986). This model is well suited to horse racing and has the convenient property that its output is a set of probability estimates which sum to 1 within each race.

> At this point, Benter is still using the multinomial logit model, which assumes that errors follow the Laplace distribution. In Multinomial Probit Models for Competitive Horse Racing[^gu03], he later replaces this with the Probit model, which assumes Normally distributed errors. His method of training the model using Maximum Likelihood Estimation (MLE) is surprisingly similar to what we now know as the list-wise approach to [Learning-to-Rank](https://en.wikipedia.org/wiki/Learning_to_rank).

The overall goal is to estimate each horse's current performance potential. "Current performance potential" being a single overall summery index of a horse's expected performance in a particular race. To construct a model to estimate current performance potential one must investigate the available data to find those variables or *factors* which have predictive significance. The profitability of the resulting betting system will be largely determined by the predictive power of the factors chosen. The odds set by the public betting yield a sophisticated estimate of the horses' win probabilities. In order for a fundamental statistical model to be able to compete effectively, it must rival the public in sophistication and comprehensiveness. Various types of factors can be classified into groups:

- Current condition:
	- performance in recent races
	- time since last race
	- recent workout data
	- age of horse

- Past performance:
	- finishing position in past races
	- lengths behind winner in past races
	- normalized times of past races

- Adjustments to past performance:
	- strength of competition in past races
	- weight carried in past races
	- jockey's contribution to past performances
	- compensation for bad luck in past races
	- compensation for advantageous or disadvantageous post position in past races

- Present race situational factors:
	- weight to be carried
	- today's jockey's ability
	- advantages or disadvantages of the assigned post position

- Preferences which could influence the horse's performance in today's race:
	- distance preference
	- surface preference (turf vs dirt)
	- condition of surface preference (wet vs dry)
	- specific track preference

More detailed discussions of fundamental handicapping can be found in the extensive popular literature on the subject (for the author's suggested references see the list in the appendix). The data needed to calculate these factors must be entered a d checked for accuracy. This can involve considerable effort. Often, multiple sources must be used to assemble complete past performance records for each of the horses. This is especially the case when the horses have run past races at many different tracks. The easiest type of racing jurisdiction to collect data and develop a model for is one with a *closed* population of horses, that is, one where horses from a single population race only against each other at a limited number of tracks. When horses have raced at venues not covered in the database, it is difficult to evaluate the elapsed times of races and to estimate the strength of their opponents. Also unknown will be the post position biases, and the relative abilities of the jockeys in those races.

In the author's experience the minimum amount of data needed for adequate model development and testing samples is in the range of 500 to 1000 races. More is helpful, but out-of-sample predictive accuracy does not seem to improve dramatically with development samples greater than 1000 races. Remember that *data for one race* means full past data on all of the runners in that race. This suggests another advantage of a *closed* racing population; by collecting the results of all races run in that jurisdiction one automatically accumulates the full set of past performance data for each horse in the population. It is important to define factors which extract as much information as possible out of the data in each of the relevant areas. As an example, consider three different specifications of a "distance preference" factor.

The first is from Bolton and Chapman (1986):

- **NEWDIST:** this variable equals one if a horse has run three of its four previous races at a distance less than a mile, zero otherwise. (Note: Bolton and Chapman's model was only used to predict races of 1-1.25 miles.)

The second is from Brecher (1980):

- **DOK:** this variable equals one if the horse finished in the upper 50th percentile or within 6.25 lengths of the winner in a prior race within 1/16 of a mile of today's distance, or zero otherwise

The last is from the author's current model:

- **DP6A:** for each of a horse's past races, a predicted finishing position is calculated via multiple regression based on all factors except those relating to distance. This predicted finishing position in each race is then subtracted from the horse's actual finishing position. The resulting quantity can be considered to be the unexplained residual which may be due to some unknown distance preference that the horse may possess plus a certain amount of random error. To estimate the horse's preference or aversion to today's distance, the residual in each of its past races is used to estimate a linear relationship between performance and similarity to today's distance. Given the statistical uncertainty of estimating this relationship from the usually small sample of past races, the final magnitude of the estimate is standardized by dividing it by its standard error. The result is that horses with a clearly defined distance preference demonstrated over a large number of races will be awarded a relatively larger magnitude value than in cases where the evidence is less clear.

> Benter later refines this idea in the paper [Modelling Distance Preference in Thoroughbred Racehorses](http://www.mathsportinternational.com/anziam/Mathsport%203.pdf)[^benter96] by adding synthetic data points when performing a constrained parabolic least squares fit for the distance preference. These "tack points" allow a solution in cases of less than three observations as well as having a regularisation effect.

The last factor is the result of a large number of progressive refinements. The subroutines involved in calculating it run to several thousand lines of code. The author's guiding principle in factor improvement has been a combination of educated guessing and trial and error. Fortunately, the historical data makes the final decision as to which particular definition is superior. The best is the one that produces the greatest increase in predictive accuracy when included in the model. The general thrust of model development is to continually experiment with refinements of the various factors. Although time-consuming, the gains are worthwhile. In the author's experience, a model involving only simplistic specifications of factors does not provide sufficiently accurate estimates of winning probabilities. Care must be taken in this process of model development not to overfit past data. Some overfitting will always occur, and for this reason it is important to use data partitioning to maintain sets of unseen races for out-of-sample testing.

The complexity of predicting horse performance makes the specification of an elegant handicapping model quite difficult. Ideally, each independent variable would capture a unique aspect of the influences effecting horse performance. In the author's experience, the trial and error method of adding independent variables to increase the model's goodness-of-fit, results in the model tending to become a hodgepodge of highly correlated variables whose individual significances are difficult to determine and often counter-intuitive. Although aesthetically unpleasing, this tendency is of little consequence for the purpose which the model will be used, namely, prediction of future race outcomes. What it does suggest, is that careful and conservative statistical tests and methods should be used on as large a data sample as possible.

For example, "number of past races" is one of the more significant factors in the author's handicapping model, and contributes greatly to the overall accuracy of the predictions. The author knows of no "common sense" reason why this factor should be important. The only reason it can be confidently included in the model is because the large data sample allows its significance to be established beyond a reasonable doubt.

Additionally, there will always be a significant amount of "inside information" in horse racing that cannot be readily included in a statistical model. Trainer's and jockey's intentions, secret workouts, whether the horse ate its breakfast, and the like, will be available to certain parties who will no doubt take advantage of it. Their betting will be reflected in the odds. This presents an obstacle to the model developer with access to published information only. For a statistical model to compete in this environment, it must make full use of the advantages of computer modelling, namely, the ability to make complex calculations on large data sets.

## Creating Unbiased Probability Estimates

It can be presumed that valid fundamental information exists which can not be systematically or practically incorporated into a statistical model. Therefore, any statistical model, however well developed, will always be incomplete. An extremely important step in model development, and one that the author believes has been generally overlooked in the literature, is the estimation of the relation of the model's probability estimates to the public's estimates, and the adjustment of the model's estimates to incorporate whatever information can be gleaned from the public's estimates. The public's implied probability estimates generally correspond well with the actual frequencies of winning. This can be shown with a table of estimated probability versus actual frequency of winning (Table 1)

<figure class="tabular-nums" markdown="1">
<figcaption>Table 1: Public Estimate vs. Actual Frequency</figcaption>

|        range |    n |  exp. |  act. |    Z |
|-------------:|-----:|------:|------:|-----:|
| 0.000--0.010 | 1343 | 0.007 | 0.007 |  0.0 |
| 0.010--0.025 | 4356 | 0.017 | 0.020 |  1.3 |
| 0.025--0.050 | 6193 | 0.037 | 0.042 |  2.1 |
| 0.050--0.100 | 8720 | 0.073 | 0.069 | -1.5 |
| 0.100--0.150 | 5395 | 0.123 | 0.125 |  0.6 |
| 0.150--0.200 | 3016 | 0.172 | 0.173 |  0.1 |
| 0.200--0.250 | 1811 | 0.222 | 0.219 | -0.3 |
| 0.250--0.300 | 1015 | 0.273 | 0.253 | -1.4 |
| 0.300--0.400 |  716 | 0.339 | 0.339 |  0.0 |
|       >0.400 |  312 | 0.467 | 0.484 |  0.6 |

<p class="text-sm text-zinc-500 dark:text-zinc-400"># races = 3198, # horses = 32877</p>
</figure>

<figure class="tabular-nums" markdown="1">
<figcaption>Table 2: Fundamental Model vs. Actual Frequency</figcaption>

|        range |    n |  exp. |  act. |    Z |
|-------------:|-----:|------:|------:|-----:|
| 0.000--0.010 | 1173 | 0.006 | 0.005 | -0.6 |
| 0.010--0.025 | 3641 | 0.018 | 0.015 | -1.2 |
| 0.025--0.050 | 6503 | 0.037 | 0.037 | -0.3 |
| 0.050--0.100 | 9642 | 0.073 | 0.074 |  0.1 |
| 0.100--0.150 | 5405 | 0.123 | 0.12  | -0.7 |
| 0.150--0.200 | 2979 | 0.173 | 0.183 |  1.6 |
| 0.200--0.250 | 1599 | 0.223 | 0.232 |  0.9 |
| 0.250--0.300 |  870 | 0.272 | 0.285 |  0.9 |
| 0.300--0.400 |  741 | 0.341 | 0.32  | -1.2 |
|       >0.400 |  324 | 0.475 | 0.432 | -1.6 |

<p class="text-sm text-zinc-500 dark:text-zinc-400"># races = 3198, # horses = 32877</p>
</figure>

**range** = the range of estimated probabilities

**n** = the number of horses falling within a range

**exp.** = the mean expected probability

**act.** = the actual win frequency observed

**Z** = the discrepancy ( + or - ) in units of standard errors

In each range of estimated probabilities, the actual frequencies correspond closely. This is not the case at all tracks (Ali, 1977) and if not, suitable corrections should be made when using the public's probability estimates for the purposes which will be discussed later. (Unless otherwise noted, data samples consist of all races run by the Royal Hong Kong Jockey Club from September 1986 through June 1993.)

> Here, Benter discusses the merits of model calibration, that estimated probabilities should occur with the same relative frequency in observed outcomes. Below, we regenerate Table 1 (Public Estimate vs. Actual Frequency) from the paper using our data for each of the four decades. Note that our table for the 1986--1993 period contains 179 additional races and it is unclear where the discrepancy lies.


```python
import numpy as np


def table(data_file, column, start, end):
    """Generate a table comparing estimated probabilities to actual win frequencies.

    Parameters:
    - data_file (str): The path to the CSV file containing the data.
    - column (str): The name of the column containing the estimated probabilities.
    - start (str): The start date for filtering the dataset.
    - end (str): The end date for filtering the dataset.

    Displays:
    - DataFrame: A table with columns: range, n, exp., act., and Z.
    - Summary: A printout of start/end date, number of races, and number of starters.
    """

    # Load the dataset from a CSV file with specified multi-index columns
    df = pd.read_csv(data_file, index_col=[0, 1, 2, 3])

    # Compute win probabilities from win odds
    df["p_overround"] = 1 / df.win_odds
    df["p"] = df.p_overround / df.groupby(["date", "venue", "number"]).p_overround.sum()

    # Filter the DataFrame based on the date range
    df = df[(df.index.get_level_values("date") > start) &
            (df.index.get_level_values("date") < end)]

    # Compute the number of unique races and the total number of starters
    races = df.groupby(["date", "venue", "number"]).ngroups
    starters = len(df)

    # Create a binary column indicating whether the horse was a winner
    df["winner"] = df.place == 1

    # Define probability bins and labels for them
    bins = [0.00, 0.01, 0.025, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.400, 1.000]
    labels = [f"{a:.3f}-{b:.3f}" for a, b in zip(bins, bins[1:])]

    # Group by the probability range and compute required aggregate values
    df = df.groupby(pd.cut(df[column].rename("range"), bins=bins, labels=labels)).agg(
        **{
            "n": (column, "size"),
            "exp.": (column, "mean"),
            "act.": ("winner", "mean"),
            "p_std": (column, "std"),
        }
    )

    # Compute the Z score to show the discrepancy in units of standard errors
    df["Z"] = (df["act."] - df["exp."]) / (df.p_std / np.sqrt(df.n)) / 25

    # Display the computed table and summary information
    display(df.drop("p_std", axis=1))
    print(f"{start=} {end=} {races=} {starters=}")


for start, end in DATE_RANGES:
    table(DATA_FILE, "p", start, end)
```

<figure class="tabular-nums" markdown="1">
<figcaption>Table B: Public Estimate vs. Actual Frequency (1986-1993)</figcaption>

|        range |    n |     exp. |     act. |         Z |
|-------------:|-----:|---------:|---------:|----------:|
| 0.000--0.010 | 1703 | 0.008470 | 0.005872 | -8.250281 |
| 0.010--0.025 | 4754 | 0.017405 | 0.019352 |  1.244934 |
| 0.025--0.050 | 6192 | 0.036615 | 0.041021 |  1.980053 |
| 0.050--0.100 | 9422 | 0.073415 | 0.069518 | -1.047902 |
| 0.100--0.150 | 5519 | 0.122698 | 0.126110 |  0.711950 |
| 0.150--0.200 | 3123 | 0.172767 | 0.174832 |  0.311883 |
| 0.200--0.250 | 1822 | 0.222284 | 0.221734 | -0.067020 |
| 0.250--0.300 | 1001 | 0.272763 | 0.254745 | -1.701837 |
| 0.300--0.400 |  790 | 0.336680 | 0.337975 |  0.052713 |
|       >0.400 |  419 | 0.478898 | 0.496420 |  0.196101 |

<p class="text-sm text-zinc-500 dark:text-zinc-400"># races = 3377, # starters = 34745</p>
</figure>


<figure class="tabular-nums" markdown="1">
<figcaption>Table C: Public Estimate vs. Actual Frequency (1996-2003)</figcaption>

|        range |    n |      exp. |     act. |         Z |
|-------------:|-----:|----------:|---------:|----------:|
| 0.000--0.010 |  5580 | 0.008393 | 0.005197 | -19.884794 |
| 0.010--0.025 |  9848 | 0.017252 | 0.017262 |   0.009684 |
| 0.025--0.050 | 11572 | 0.036291 | 0.036467 |   0.108673 |
| 0.050--0.100 | 15455 | 0.072603 | 0.072986 |   0.132493 |
| 0.100--0.150 |  7978 | 0.122074 | 0.127852 |   1.467941 |
| 0.150--0.200 |  3726 | 0.172453 | 0.163983 |  -1.418650 |
| 0.200--0.250 |  1883 | 0.222604 | 0.222517 |  -0.010583 |
| 0.250--0.300 |  1001 | 0.272983 | 0.299700 |   2.473131 |
| 0.300--0.400 |   836 | 0.337409 | 0.330144 |  -0.308628 |
|       >0.400 |   393 | 0.468867 | 0.424936 |  -0.508843 |

<p class="text-sm text-zinc-500 dark:text-zinc-400"># races = 4534, # starters = 58272</p>
</figure>

<figure class="tabular-nums" markdown="1">
<figcaption>Table D: Public Estimate vs. Actual Frequency (2006-2013)</figcaption>

|        range |     n |     exp. |     act. |          Z |
|-------------:|------:|---------:|---------:|-----------:|
| 0.000--0.010 |  6367 | 0.008398 | 0.004869 | -23.329629 |
| 0.010--0.025 | 11433 | 0.017268 | 0.015219 |  -2.002755 |
| 0.025--0.050 | 13139 | 0.036298 | 0.036609 |   0.204394 |
| 0.050--0.100 | 17404 | 0.072359 | 0.072052 |  -0.113371 |
| 0.100--0.150 |  8199 | 0.121985 | 0.120746 |  -0.314975 |
| 0.150--0.200 |  4095 | 0.172668 | 0.179243 |   1.144066 |
| 0.200--0.250 |  2180 | 0.223207 | 0.229817 |   0.866324 |
| 0.250--0.300 |  1296 | 0.272618 | 0.273920 |   0.134999 |
| 0.300--0.400 |  1207 | 0.339081 | 0.345485 |   0.325636 |
|       >0.400 |   672 | 0.472380 | 0.497024 |   0.391450 |

<p class="text-sm text-zinc-500 dark:text-zinc-400"># races = 5261, # starters = 65992</p>
</figure>

<figure class="tabular-nums" markdown="1">
<figcaption>Table E: Public Estimate vs. Actual Frequency (2016-2023)</figcaption>

|        range |     n |     exp. |     act. |         Z |
|-------------:|------:|---------:|---------:|----------:|
| 0.000--0.010 |  8135 | 0.006063 | 0.004179 | -3.080125 |
| 0.010--0.025 | 11967 | 0.017164 | 0.015961 | -1.216143 |
| 0.025--0.050 | 12685 | 0.036488 | 0.033819 | -1.709530 |
| 0.050--0.100 | 17920 | 0.072736 | 0.069420 | -1.248783 |
| 0.100--0.150 |  8340 | 0.122367 | 0.126859 |  1.141811 |
| 0.150--0.200 |  4536 | 0.172814 | 0.177028 |  0.771925 |
| 0.200--0.250 |  2584 | 0.224027 | 0.229102 |  0.724819 |
| 0.250--0.300 |  1520 | 0.272469 | 0.276974 |  0.518236 |
| 0.300--0.400 |  1519 | 0.340612 | 0.356814 |  0.915666 |
|       >0.400 |   865 | 0.486887 | 0.528324 |  0.621833 |

<p class="text-sm text-zinc-500 dark:text-zinc-400"># races = 5757, # starters = 70071</p>
</figure>

A multinomial logit model using fundamental factors will also naturally produce an internally consistent set of probability estimates (Table 2). Here again there is generally good correspondence between estimated and actual frequencies. Table 2 however conceals a major, (and from a wagering point of view, disastrous) type of bias inherent in the fundamental model's probabilities. Consider the following two tables which represent roughly equal halves of the sample in Table 2. Table 3 shows the fundamental model's estimate versus actual frequency for those horses where the public's probability estimate was greater the fundamental model's. Table 4 is the same except that it is for those horses whose public estimate was less than the fundamental model's.

<figure class="tabular-nums" markdown="1">
<figcaption>Table 3: Fundamental Model vs. Actual Frequency When Public Estimate Is Greater Than Model Estimate</figcaption>

|        range |    n |  exp. |  act. |    Z |
|-------------:|-----:|------:|------:|-----:|
| 0.000--0.010 |  920 | 0.006 | 0.005 | -0.3 |
| 0.010--0.025 | 2130 | 0.017 | 0.018 |  0.3 |
| 0.025--0.050 | 3454 | 0.037 | 0.044 |  2.1 |
| 0.050--0.100 | 4626 | 0.073 | 0.091 |  4.7 |
| 0.100--0.150 | 2413 | 0.122 | 0.147 |  3.7 |
| 0.150--0.200 | 1187 | 0.172 | 0.227 |  5.0 |
| 0.200--0.250 |  540 | 0.223 | 0.302 |  4.4 |
| 0.250--0.300 |  252 | 0.270 | 0.333 |  2.3 |
| 0.300--0.400 |  165 | 0.342 | 0.448 |  2.9 |
|       >0.400 |   54 | 0.453 | 0.519 |  1.0 |

<p class="text-sm text-zinc-500 dark:text-zinc-400"># races = 3198, # horses = 15741</p>
</figure>

<figure class="tabular-nums" markdown="1">
<figcaption>Table 4: Fundamental Model vs. Actual Frequency When Public Estimate Is Less Than Model Estimate</figcaption>

|        range |    n |  exp. |  act. |    Z |
|-------------:|-----:|------:|------:|-----:|
| 0.000--0.010 |  253 | 0.007 | 0.004 | -0.6 |
| 0.010--0.025 | 1511 | 0.018 | 0.011 | -2.2 |
| 0.025--0.050 | 3049 | 0.037 | 0.029 | -2.6 |
| 0.050--0.100 | 5016 | 0.074 | 0.058 | -4.3 |
| 0.100--0.150 | 2992 | 0.123 | 0.098 | -4.2 |
| 0.150--0.200 | 1792 | 0.173 | 0.154 | -2.1 |
| 0.200--0.250 | 1059 | 0.223 | 0.196 | -2.1 |
| 0.250--0.300 |  618 | 0.273 | 0.265 | -0.4 |
| 0.300--0.400 |  576 | 0.341 | 0.283 | -2.9 |
|       >0.400 |  270 | 0.480 | 0.415 | -2.1 |

<p class="text-sm text-zinc-500 dark:text-zinc-400"># races = 3198, # horses = 17136</p>
</figure>

There is an extreme and consistent bias in both tables. In virtually every range the actual frequency is significantly different than the fundamental model's estimate, and always in the direction of being closer to the public's estimate. The fundamental model's estimate of the probability cannot be considered to be an unbiased estimate independent of the public's estimate. Table 4 is particularly important because it is comprised of those horses that the model would have one bet on, that is, horses whose model-estimated probability is greater than their public probability. It is necessary to correct for this bias in order to accurately estimate the advantage of any particular bet.'

In a sense, what is needed is a way to combine the judgements of two experts, (i.e. the fundamental model and the public). One practical technique for accomplishing this is as follows: (Asch and Quandt, 1986; pp. 123--125). See also White, Dattero and Flores, (1992).

Estimate a second logit model using the two probability estimates as independent variables. For a race with entrants $$(1.2, ..., N)$$ the win probability of horse $$i$$ is given by:

$$c_i = \frac{\exp(\alpha f_i + \beta \pi_i)}{\sum_{j=1}^{N} \exp (\alpha f_j + \beta \pi_j)}$$

$$f_i$$ = log of "out-of-sample" fundamental model probability estimate

$$\pi_i$$ = log of public's implied probability estimate

$$c_i$$ = combined probability estimate

(Natural log of probability is used rather than probability as this transformation provides a better fit)

```python
import lightning.pytorch as pl

from torchmetrics.functional.classification import multiclass_accuracy


class CombinedModel(pl.LightningModule):
    def __init__(self, num_features: int, places: int = 4, lr: float = 1e-3):
        super().__init__()
        self.example_input_array = torch.rand(32, num_features), torch.rand(32, 1)
        self.save_hyperparameters()

        self.fundamental = nn.Linear(num_features, 1, bias=False)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.adjustments = nn.Parameter(torch.ones(places - 1))

    def forward(self, x, p):
        fundamental_log_probs = F.log_softmax(self.fundamental(x).squeeze(), dim=-1)
        public_log_probs = torch.log(p)
        logits = self.alpha * fundamental_log_probs + self.beta * public_log_probs
        return softmax(logits, self.adjustments, self.hparams.places)

    def training_step(self, batch, batch_idx):
        loss, _ = self._shared_evaluation(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_evaluation(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        return metrics

    def _shared_evaluation(self, batch, batch_idx):
        x, p, y = batch
        y_hat = self(x, p)
        loss = harville_loss(y_hat, y, self.hparams.places)
        acc = multiclass_accuracy(y_hat[:,:,0].argmax(dim=-1), y.argmin(dim=-1), 14)
        return loss, acc

    def predict_step(self, batch, batch_idx):
        x, p, y = batch
        return self(x, p), p, y

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
```

Given a set of past races $$(1, 2, ..., R)$$, for which both public probability estimates and fundamental model estimates are available, the parameters $$\alpha$$ and $$\beta$$ can be estimated by maximizing the log likelihood function of the given set of races with respect to $$\alpha$$ and $$\beta$$:

$$\exp(L) = \prod_{j-1}^{R} c_{ji*}$$

where $$c_{ji*}$$ denotes the probability as given by equation (1) for the horse $$i*$$ observed to win race $$j$$ (Bolton and Chapman, 1986 p. 1044). Equation (1) should be evaluated using fundamental probability estimates from a model developed on a separate sample of races. Use of "out-of-sample" estimates prevents overestimation of the fundamental model's significance due to "custom-fitting" of the model development sample. The estimated values of $$\alpha$$ and $$\beta$$ can be interpreted roughly as the relative correctness of the model's and the public's estimates. The greater the value of a,the better the model. The probabilities that result from this model also show good correspondence between predicted and actual frequencies of winning (Table 5).

<figure class="tabular-nums" markdown="1">
<figcaption>Table 5: Combined Model vs. Actual Frequencies</figcaption>

|        range |    n |  exp. |  act. |    Z |
|-------------:|-----:|------:|------:|-----:|
| 0.000--0.010 | 1520 | 0.007 | 0.005 | -1.0 |
| 0.010--0.025 | 4309 | 0.017 | 0.018 |  0.1 |
| 0.025--0.050 | 6362 | 0.037 | 0.038 |  0.6 |
| 0.050--0.100 | 8732 | 0.073 | 0.071 | -0.5 |
| 0.100--0.150 | 5119 | 0.123 | 0.119 | -0.8 |
| 0.150--0.200 | 2974 | 0.173 | 0.18  |  1.0 |
| 0.200--0.250 | 1657 | 0.223 | 0.223 |  0.0 |
| 0.250--0.300 |  993 | 0.272 | 0.281 |  0.6 |
| 0.300--0.400 |  853 | 0.34  | 0.328 |  0.7 |
|       >0.400 | 358  | 0.479 | 0.492 |  0.5 |

<p class="text-sm text-zinc-500 dark:text-zinc-400"># races = 3198, # horses = 32877</p>
</figure>

By comparison with Tables 1 and 2, Table 5 shows that there is more *spread* in the combined model's probabilities than in either the public's or the fundamental model's alone, that is, there are more horses in both the very high and very low probability ranges. This indicates that the combined model is more informative. More important is that the new probability estimates are without the bias shown in Tables 3 and 4, and thus are suitable for the accurate estimation of betting advantage. This is borne out by Tables 6 and 7, which are analogous to Tables 3 and 4 above except that they use the combined model probabilities instead of the raw fundamental model probabilities. 

<figure class="tabular-nums" markdown="1">
<figcaption>Table 6: Combined Model vs. Actual Frequency When Public Estimate Is Greater Than Model Estimate</figcaption>

|        range |    n |  exp. |  act. |    Z |
|-------------:|-----:|------:|------:|-----:|
| 0.000--0.010 |  778 | 0.006 | 0.005 | -0.4 |
| 0.010--0.025 | 1811 | 0.017 | 0.015 | -0.6 |
| 0.025--0.050 | 2874 | 0.037 | 0.035 | -0.7 |
| 0.050--0.100 | 4221 | 0.073 | 0.073 |  0.0 |
| 0.100--0.150 | 2620 | 0.123 | 0.116 | -1.0 |
| 0.150--0.200 | 1548 | 0.173 | 0.185 |  1.2 |
| 0.200--0.250 |  844 | 0.223 | 0.231 |  0.6 |
| 0.250--0.300 |  493 | 0.272 | 0.292 |  1.0 |
| 0.300--0.400 |  393 | 0.337 | 0.349 |  0.5 |
|       >0.400 |  159 | 0.471 | 0.509 |  1.0 |

<p class="text-sm text-zinc-500 dark:text-zinc-400"># races = 3198, # horses = 15741</p>
</figure>

<figure class="tabular-nums" markdown="1">
<figcaption>Table 7: Combined Model vs. Actual Frequency When Public Estimate Is Less Than Model Estimate</figcaption>

|        range |    n |  exp. |  act. |    Z |
|-------------:|-----:|------:|------:|-----:|
| 0.000--0.010 |  742 | 0.007 | 0.004 | -0.9 |
| 0.010--0.025 | 2498 | 0.018 | 0.019 |  0.6 |
| 0.025--0.050 | 3488 | 0.037 | 0.041 |  1.4 |
| 0.050--0.100 | 4511 | 0.072 | 0.069 | -0.7 |
| 0.100--0.150 | 2499 | 0.123 | 0.122 | -0.1 |
| 0.150--0.200 | 1426 | 0.173 | 0.174 |  0.1 |
| 0.200--0.250 |  813 | 0.223 | 0.215 | -0.5 |
| 0.250--0.300 |  500 | 0.272 | 0.270 | -0.1 |
| 0.300--0.400 |  460 | 0.342 | 0.311 | -1.4 |
|       >0.400 |  199 | 0.485 | 0.477 | -0.2 |

<p class="text-sm text-zinc-500 dark:text-zinc-400"># races = 3198, # horses = 17136</p>
</figure>

Observe that the above tables show no significant bias one way or the other.

## Assessing the Value of a Handicapping Model

The log likelihood function of equation (2) can be used to produce a measure of fit analogous to the R<sup>2</sup> of multiple linear regression (Equation 3). This pseudo-R<sup>2</sup> $$(R^2)$$ can be used to compare models and to assess the value of a particular model as a betting tool. Each set of probability estimates, either the public's or those of a model, achieve a certain $$R^2$$, defined as (Bolton and Chapman, 1986)

$$R^2 = 1 - \frac{L(\textrm{model})}{L(1/N_j)}$$

The $$R^2$$ value is a measure of the "explanatory power" of the model. An $$R^2$$ of 1 indicates perfect predictive ability while an $$R^2$$ of 0 means that the model is no better than random guessing. An important benchmark is the $$R^2$$ value achieved by the public probability estimate. A heuristic measure of the potential profitability of a handicapping model, borne out in practice, is the amount by which its inclusion in the combined model of equation (1) along with the public probability estimate causes the $$R^2$$ to increase over the value achieved by the public estimate alone:

$$\Delta R^2 = R^2_{\textrm{C}} - R^2_{\textrm{P}}$$

where the subscript $$\textrm{P}$$ denotes the public's probability estimate and $$\textrm{C}$$ stands for the combined (fundamental and public) model of equation (1) above. In a sense $$\Delta R^2$$ may be taken as a measure of the amount of information added by the fundamental model. In the case of the models which produced Tables 1, 2 and 5 above these values are:

$$
\begin{aligned}
R^2_{\textrm{P}} &= 0.1218 \ \textrm{(public)}\\

R^2_{\textrm{F}} &= 0.1245 \ \textrm{(fundamental model)}\\

R^2_{\textrm{C}} &= 0.1396 \ \textrm{(combined model)}\\

\Delta R^2_{\textrm{C} \cdot \textrm{P}} &= 0.1396 - 0.1218 = 0.0178\\
\end{aligned}
$$

Though this value may appear small, it actually indicates that significant profits could be made with that model. The $$\Delta R^2$$ value is a useful measure of the potential profitability of a particular model. It can be used to measure and compare models without the the time consuming step of a full wagering simulation. In the author's experience, greater $$\Delta R^2$$ values have been invariably associated with greater wagering simulation profitability. To illustrate the point that the important criteria is the gain in $$R^2$$ in the combined model over the public's $$R^2$$, and not simply the $$R^2$$ of the handicapping model alone, consider the following two models.

The first is a logit-derived fundamental handicapping model using 9 significant fundamental factors. It achieves an out-of-sample $$R^2$$ of 0.1016. The second is a probability estimate derived from tallying the picks of approximately 48 newspaper tipsters. (Figlewski, 1979) The tipsters each make a selection for 1st, 2nd, and 3rd in each race. The procedure was to count the number of times each horse was picked, awarding 6 points for 1st, 3 points for 2nd and 1 point for 3rd. The point total for each horse is then divided by the total points awarded in the race (i.e. 48 * 10). This fraction of points is then taken to be the "tipster" probability estimate. Using the log of this estimate as the sole independent variable in a logit model produces an $$R^2$$ of 0.1014. On the basis of their stand-alone $$R^2$$'s the above two models would appear to be equivalently informative predictors of race outcome. Their vast difference appears when we perform the 'second stage' of combining these estimates with the public's. The following results were derived from logit runs on 2,313 races (September 1988 to June 1993).

$$
\begin{aligned}
R^2_{\textrm{P}} &= 0.1237 \ \textrm{(public estimate)}\\

R^2_{\textrm{F}} &= 0.1016 \ \textrm{(fundamental model)}\\

R^2_{\textrm{T}} &= 0.1014 \ \textrm{(tipster model)}\\

R^2_{(\textrm{F} \& \textrm{P}) - \textrm{P}} &= 0.1327 \ \textrm{(fundamental and public)}\\

R^2_{(\textrm{T} \& \textrm{P}) - \textrm{P}} &= 0.1239 \ \textrm{(tipster and public)}\\

\Delta R^2_{(\textrm{F} \& \textrm{P}) - \textrm{P}} &= 0.1327 - 0.1237 = 0.0090\\

\Delta R^2_{(\textrm{T} \& \textrm{P}) - \textrm{P}} &= 0.1239 - 0.1237 = 0.0002\\
\end{aligned}
$$

As indicated by the $$\Delta R^2$$ values, the tipster model adds very little to the public's estimate. The insignificant contribution of the tipster estimate to the overall explanatory power of the combined model effectively means that when there is a difference between the public estimate and the tipster estimate, then the public's estimate is superior. The fundamental model on the other hand, does contribute significantly when combined with the public's. For a player considering betting with the "tipster" model, carrying out this "second stage" would have saved that player from losing money; the output of the second stage model would always be virtually identical to the public estimate, thus never indicating an advantage bet.

> While we may use negative log likelihood loss when training a fundamental model, Benter's pseudo-R<sup>2</sup> metric is easy to interpret. We can see that with each subsequent decade, the public estimate improves.


```python
import torch
import torch.nn.functional as F

from torch import nn, optim, Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data


def dataset(data_file: str, start_date: str, end_date: str):
    """
    Load a horse race dataset, filter by date range, and convert to pytorch tensors.

    Parameters:
    - data_file (str): The path to the CSV file containing the data.
    - start (str): The start date for filtering the dataset.
    - end (str): The end date for filtering the dataset.

    Returns:
    - TensorDataset: A pytorch TensorDataset containing padded sequences for
      probabilities (p), and places (y).
    """

    # Load the dataset from a CSV file with specified multi-index columns
    df = pd.read_csv(data_file, index_col=[0, 1, 2, 3])
    
    # Compute win probabilities from win odds
    df["p_overround"] = 1 / df.win_odds
    df["p"] = df.p_overround / df.groupby(["date", "venue", "number"]).p_overround.sum()

    # Filter the DataFrame based on the date range
    df = df[(df.index.get_level_values("date") >= start) &
            (df.index.get_level_values("date") < end)]

    # Separate probabilities (p), and places (y)
    p = df.p
    y = df.place

    # Group by date, venue, and race number, and convert each group to a torch tensor
    p_race = p.groupby(["date", "venue", "number"], observed=True).apply(
        lambda df: torch.tensor(df.values)
    )
    y_race = y.groupby(["date", "venue", "number"], observed=True).apply(
        lambda df: torch.tensor(df.values)
    )

    # Pad the sequences for each group to have the same length
    p_pad = pad_sequence(tuple(p_race), batch_first=True, padding_value=1e-9).float()
    y_pad = pad_sequence(tuple(y_race), batch_first=True, padding_value=999).long()

    # Return as a pytorch TensorDataset
    return data.TensorDataset(p_pad, y_pad)
```


```python
def pseudo_r2(probs: Tensor, targets: Tensor) -> Tensor:
    """
    Calculate the pseudo R-squared metric for a horse race betting model.

    Parameters:
    - probs (Tensor): A tensor containing the predicted probabilities for each horse in each race.
    - targets (Tensor): A tensor containing the true finishing positions for each horse in each race.

    Returns:
    - Tensor: A tensor containing the pseudo R-squared value.

    Note: This function assumes that targets with values >= 100 are not participating in the race.
    """
    
    # Count the number of horses participating in each race
    horses_per_race = (targets < 100).sum(axis=1)
    
    # Generate random noise to break dead heats
    noise = torch.rand(targets.shape).to(targets.device)
    
    # Sort targets and get the index of the winning horse
    win_index = torch.argsort(targets + noise)[:, 0]
    
    # Extract the predicted probabilities of the winners
    win_probs = torch.gather(probs, 1, win_index.unsqueeze(-1))
    
    # Calculate log-likelihood for the model
    L_model = torch.log(win_probs).sum()
    
    # Calculate log-likelihood under a random model
    L_random = torch.log(1 / horses_per_race).sum()
    
    # Calculate the pseudo R-squared value
    r2 = 1 - L_model / L_random
    
    return r2
```


```python
for start, end in DATE_RANGES:
    p_pad, y_pad = dataset(DATA_FILE, start, end)[:]
    r2 = pseudo_r2(p_pad, y_pad).item()
    print(f"Public Estimate ({start[:4]}-{end[:4]} Seasons): r2={r2:.4f}")
```

    Public Estimate (1986-1993 Seasons)
    r2=0.1325
    
    Public Estimate (1996-2003 Seasons)
    r2=0.1437
    
    Public Estimate (2006-2013 Seasons)
    r2=0.1668
    
    Public Estimate (2016-2023 Seasons)
    r2=0.1863


## Wagering Strategy

After computing the combined and therefore unbiased probability estimates as described above, one can make accurate estimations of the advantage of any particular bet. A way of expressing advantage is as the expected return per dollar bet:

$$\textrm{expected return} = er = c * div$$

$$\textrm{advantage} = er - 1$$

where $$c$$ is the estimated probability of winning the bet and $$div$$ is the expected dividend. For win betting the situation is straightforward. The $$c$$'s are the probability estimates produced by equation (1) above, and the $$div$$'s are the win dividends (as a payoff for a $1 bet) displayed on the tote board. The situation for an example race is illustrated in Table 8.

<figure class="tabular-nums" markdown="1">
<figcaption>Table 8</figcaption>

|  # |     c |     p |   er |  div |
|---:|------:|------:|-----:|-----:|
|  1 | 0.021 | 0.025 | 0.68 | 33.0 |
|  2 | 0.125 | 0.088 | 1.17 |  9.3 |
|  3 | 0.239 | 0.289 | 0.69 |  2.8 |
|  4 | 0.141 | 0.134 | 0.87 |  6.1 |
|  5 | 0.066 | 0.042 | 1.29 | 19.0 |
|  6 | 0.012 | 0.013 | 0.75 | 61.0 |
|  7 | 0.107 | 0.136 | 0.64 |  6.0 |
|  8 | 0.144 | 0.089 | 1.33 |  9.2 |
|  9 | 0.019 | 0.014 | 1.18 | 60.0 |
| 10 | 0.067 | 0.066 | 0.68 | 12.0 |
| 11 | 0.012 | 0.012 | 0.83 | 68.0u|
| 12 | 0.028 | 0.047 | 0.50 | 17.0 |
| 13 | 0.011 | 0.027 | 0.32 | 30.0 |
| 14 | 0.009 | 0.019 | 0.41 | 43.0 |

</figure>

**c** = combined (second stage) probability estimate

**p** = public's probability estimate (1-take) / div

**er** = expected return on a $1 win bet

**div** = win dividend for a $1 bet

The "u" after the win dividend for horse #11 stands for *unratable* and indicates that this is a horse for which the fundamental model could not produce a probability estimate. Often this is because the horse is running in its first race. A good procedure for handling such horses is to assign them the same probability as that implied by the public win odds, and renormalize the probabilities on the other horses so that the total probability for the race sums to 1, This is equivalent to saying that we have no information which would allow us to dispute the public's estimate so we will take theirs.

From Table 8 we can see that the advantage win bets are those with an er greater than 1. There is a positive expected return from betting on each of these horses. Given that there are several different types of wager available, it is necessary to have a strategy for determining which bets to make and in what amounts.

## Kelly Betting and Pool Size Limitations

Given the high cost in time and effort of developing a winning handicapping system, a wagering strategy which produces maximum expected profits is desirable. The stochastic nature of horse race wagering however, guarantees that losing streaks of various durations will occur. Therefore a strategy which balances the tradeoff between risk and returns is necessary. A solution to this problem is provided by the Kelly betting strategy (Kelly, 1956). The Kelly strategy specifies the fraction of total wealth to wager so as to maximize the exponential rate of growth of wealth, in situations where the advantage and payoff odds are known. As a fixed fraction strategy, it also never risks ruin. (This last point is not strictly true, as the minimum bet limit prevents strict adherence to the strategy.) For a more complete discussion of the properties of the Kelly strategy see MacLean, Ziemba and Blazenko (1992), see also Epstein (1977) and Brecher (1980).

The Kelly strategy defines the optimal bet (or set of bets) as those which maximize the expected log of wealth. In pari-mutuel wagering, where multiple bets are available in each race, and each bet effects the final payoff odds,the exact solution requires maximizing a concave logarithmic function of several variables. For a single bet, assuming no effect on the payoff odds, the formula simplifies to

$$K = \frac{\textrm{advantage}}{\textrm{dividend} - 1}$$

where $$K$$ is the fraction of total wealth to wager, When one is simultaneously making wagers in multiple pools, further complications to the exact multiple bet Kelly solution arise due to "exotic" bets in which one must specify the order of finish in two or more races. The expected returns from these bets must be taken into account when calculating bets for the single race pools in those races.

In the author's experience, betting the full amount recommended by the Kelly formula is unwise for a number of reasons. Firstly, accurate estimation of the advantage of the bets is critical; if one overestimates the advantage by more than a factor of two, Kelly betting will cause a negative rate of capital growth. (As a practical matter, many factors may cause one's real-time advantage to be less than past simulations would suggest, and very few can cause it to be greater. Overestimating the advantage by a factor of two is easily done in practice.) Secondly, if it is known that regular withdrawals from the betting bankroll will be made for paying expenses or taking profits, then one's effective wealth is less than their actual current wealth. Thirdly, full Kelly betting is a "rough ride", downswings during which more than 50% of total wealth is lost are a common occurrence. For these and other reasons, *fractional* Kelly betting strategy is advisable, that is, a strategy wherein one bets some fraction of the recommended Kelly bet (e.g. 1/2 or 1/3), For further discussion of fractional Kelly betting, and a quantitative analysis of the risk/reward tradeoffs involved, see MacLean, Ziemba and Blazenko (1992).

Another even more important constraint on betting is the effect that one's bet has on the advantage. In pari-mutuel betting markets each bet decreases the dividend. Even if the bettor possesses infinite wealth, there is a maximum bet producing the greatest expected profit, any amount beyond which lowers the expected profit. The maximum bet can be calculated by writing the equation for expected profit as a function of bet size, and solving for the bet size which maximizes expected profit. This maximum can be surprisingly low as the following example illustrates.

<figure class="tabular-nums" markdown="1">

| c  | div | er   |
|----|-----|------|
| 06 | 20  | 1.20 |

</figure>

$$
\begin{aligned}
\textrm{total pool size} &= \$100,000\\

\textrm{maximum } er \textrm{ bet} &= \$416\\

\textrm{expected profit} &= \$39.60\\
\end{aligned}
$$

A further consideration concerns the shape of the "expected profit versus bet size" curve when the bet size is approaching the maximum. In this example, the maximum expected profit is with a bet of $416. If one made a bet of only 2/3 the maximum, i.e. $277, the expected profit would be 35.5 dollars, or 90% of the maximum. There is very little additional gain for risking a much larger sum of money. Solving the fully formulated Kelly model (i.e. taking into account the bets' effects on the dividends) will optimally balance this tradeoff. See Kallberg and Ziemba (1994) for a discussion of the optimization properties of such formulations.

As a practical matter; given the relatively small sizes of most pari-mutuel pools, a successful betting operation will soon find that all of its bets are *pool-size-limited*. As a rule of thumb, as the bettor's wealth approaches the total pool size, the dominant factor limiting bet size becomes the effect of the bet on the dividend, not the bettor's wealth.

> In Chapter 7 of [Precision: Statistical and Mathematical Methods in Horse Racing](https://www.amazon.com/Precision-Statistical-Mathematical-Methods-Racing/dp/1432768522)[^wong11], CX Wong discusses the strategic implication of the Kelly Criterion, particularly the common misconception that only the horse with the highest advantage should be bet on, with the Kelly Criterion only used to size the bet. In fact, we should bet on all overlays in a race with varying amounts, with the intuition being that we should trade-off a little return for an increased chance of winning.

### Exotic Bets

In addition to win bets, racetracks offer numerous so-called *exotic* bets. These offer some of the highest advantage wagering opportunities. This results from the multiplicative effect on overall advantage of combining more than one advantage horse. For example, suppose that in a particular race there are two horses for which the model's estimate of the win probability is greater than the public's, though not enough so as to make them positive expectation win bets.

<figure class="tabular-nums" markdown="1">

| c     |   div |     p |    er |
|-------|-------|-------|-------|
| 0.115 | 8.3   | 0.100 | 0.955 |
| 0.060 | 16.6  | 0.050 | 0.996 |

</figure>

By the Harville formula (Harville 1973), the estimated probability of a 1,2 or 2,1 finish is

$$
\begin{aligned}
C_{12,21} =& (0.115 * 0.060) / (1 - 0.115) + \\
           & (0.060 * 0.115) / (1 - 0.060)   \\
          =& 0.0151
\end{aligned}
$$

The public's implied probability estimate is

$$
\begin{aligned}
P_{12,21} =& (0.100 * 0.050) / (1 - 0.100) + \\
           & (0.050 * 0.100) / (1 - 0.050)   \\
          =& 0.0108
\end{aligned}
$$

Therefore (assuming a 17% track take) the public's rational quinella dividend should be

$$\textrm{qdiv} \cong (1 - 0.17) / 0.0108 = 76.85$$

Assuming that the estimated probability is correct the expected return of a bet on this combination is

$$er = 0.0151 * 76.85 = 1.16$$

In the above example two horses which had expected returns of less than 1 as individual win bets, in combination produce a 16% advantage quinella bet. The same principle applies, only more so, for bets in which one must specify the finishing positions of more than two horses. In *ultra-exotic* bets such as the pick-six, even a handicapping model with only modest predictive ability can produce advantage bets. The situation may be roughly summarized by stating that for a bettor in possession of accurate probability estimates which differ from the public estimates; "the more *exotic* (i.e. specific) the bet, the higher the advantage". Place and show bets are not considered exotic in this sense as they are less specific than normal bets. The probability differences are "watered down" in the place and show pools. Some professional players make only exotic wagers to capitalize on this effect.

### First, Second, and Third

In exotic bets that involve specifying the finishing order of two or more horses in one race, a method is needed to estimate these probabilities. A popular approach is the Harville formula. (Handle, 1973):

For three horses $$(i, j, k)$$ with win probabilities $$($$ the Harville formula specifies the probability that they will finish in order as

$$\pi_{ijk} = \frac{\pi_i \pi_j \pi_k}{(1 - \pi_i) (1 - \pi_i - \pi_j)}$$

This formula is significantly biased, and should not be used for betting purposes, as it will lead to serious errors in probability estimations if not corrected for in some way.' (Henery 1981, Stem 1990, Lo and Bacon-Shone 1992). Its principle deficiency is the fact that it does not recognize the increasing randomness of the contests for second and third place. The bias in the Harville formula is demonstrated in Tables 9 and 10 which show the formula's estimated probabilities for horses to finish second and third given that the identity of the horses finishing first (and second) are known. The data set used is the same as that which produced Table 1 above.

<figure class="tabular-nums" markdown="1">
<figcaption>Table 9: Harville Model Conditional Probability of 2nd</figcaption>

|        range |    n |  exp. |  act. |    Z |
|-------------:|-----:|------:|------:|-----:|
| 0.000--0.010 |  962 | 0.007 | 0.010 |  0.9 |
| 0.010--0.025 | 3449 | 0.018 | 0.030 |  5.3 |
| 0.025--0.050 | 5253 | 0.037 | 0.045 |  2.8 |
| 0.050--0.100 | 7682 | 0.073 | 0.080 |  2.3 |
| 0.100--0.150 | 4957 | 0.123 | 0.132 |  1.9 |
| 0.150--0.200 | 3023 | 0.173 | 0.161 | -1.8 |
| 0.200--0.250 | 1834 | 0.223 | 0.195 | -3.0 |
| 0.250--0.300 | 1113 | 0.272 | 0.243 | -2.3 |
| 0.300--0.400 | 1011 | 0.338 | 0.317 | -1.4 |
|       >0.400 |  395 | 0.476 | 0.372 | -4.3 |

<p class="text-sm text-zinc-500 dark:text-zinc-400"># races = 3198, # horses = 29679</p>
</figure>

<figure class="tabular-nums" markdown="1">
<figcaption>Table 10: Harville Model Conditional Probability of 3rd</figcaption>

|        range |    n |  exp. |  act. |    Z |
|-------------:|-----:|------:|------:|-----:|
| 0.000--0.010 |  660 | 0.007 | 0.009 |  0.5 |
| 0.010--0.025 | 2680 | 0.018 | 0.033 |  4.3 |
| 0.025--0.050 | 4347 | 0.037 | 0.062 |  6.8 |
| 0.050--0.100 | 6646 | 0.073 | 0.087 |  4.0 |
| 0.100--0.150 | 4325 | 0.123 | 0.136 |  2.5 |
| 0.150--0.200 | 2923 | 0.173 | 0.178 |  0.7 |
| 0.200--0.250 | 1831 | 0.223 | 0.192 | -3.4 |
| 0.250--0.300 | 1249 | 0.273 | 0.213 | -4.9 |
| 0.300--0.400 | 1219 | 0.341 | 0.273 | -5.3 |
|       >0.400 |  601 | 0.492 | 0.333 | -8.3 |

<p class="text-sm text-zinc-500 dark:text-zinc-400"># races = 3198, # horses = 26481</p>
</figure>

The large values of the Z-statistics show the significance of the bias in the Harville formula. The tendency is for low probability horses to finish second and third more often than predicted, and for high probability horses to finish second and third less often. The effect is more pronounced for 3rd place than for 2nd. An effective, and computationally economical way to correct for this is as follows: Given the win probability array, ($$\pi_1, \pi_2, ..., \pi_N)$$, create a second array $$\sigma$$ such that,

$$\sigma_i = \frac{\exp(\gamma \log(\pi_i))}{\sum_{j=1}^{N} \exp(\gamma \log(\pi_j))}$$

and a third array $$\tau$$ such that,

$$\tau_i = \frac{\exp(\delta \log(\pi_i))}{\sum_{j=1}^{N} \exp(\delta \log(\pi_j))}$$

The probability of the three horses ($$i, j, k)$$ finishing in order is then

$$\pi_{ijk} = \frac{\pi_i \sigma_j \tau_k}{(1 - \sigma_i) (1 - \tau_i - \tau_j)}$$

The parameters $$\gamma$$ and $$\delta$$ can be estimated via maximum likelihood estimation on a sample of past races. For the above data set the maximum likelihood values of the parameters are $$\gamma = 0.81$$ and $$\delta = 0.65$$. Reproducing Tables 9 and 10 above using equations (7--9) with these parameter values substantially corrects for the Harville formula bias as can be seen in Tables 11 and 12.

<figure class="tabular-nums" markdown="1">
<figcaption>Table 11: Logistic Model Conditional Probability of 2nd \((\gamma = 0.81)\)</figcaption>

|        range |    n |  exp. |  act. |    Z |
|-------------:|-----:|------:|------:|-----:|
| 0.000--0.010 |  251 | 0.008 | 0.012 |  0.6 |
| 0.010--0.025 | 2282 | 0.018 | 0.024 |  1.9 |
| 0.025--0.050 | 5195 | 0.037 | 0.033 | -1.6 |
| 0.050--0.100 | 8819 | 0.074 | 0.073 | -0.4 |
| 0.100--0.150 | 6054 | 0.123 | 0.125 |  0.5 |
| 0.150--0.200 | 3388 | 0.173 | 0.176 |  0.5 |
| 0.200--0.250 | 1927 | 0.222 | 0.216 | -0.8 |
| 0.250--0.300 |  973 | 0.272 | 0.275 |  0.2 |
| 0.300--0.400 |  616 | 0.336 | 0.349 |  0.7 |
|       >0.400 |  174 | 0.456 | 0.397 | -1.6 |

<p class="text-sm text-zinc-500 dark:text-zinc-400"># races = 3198, # horses = 29679</p>
</figure>

<figure class="tabular-nums" markdown="1">
<figcaption>Table 12: Logistic Model Conditional Probability of 3rd \((\delta = 0.65)\)</figcaption>

|        range |    n |  exp. |  act. |    Z |
|-------------:|-----:|------:|------:|-----:|
| 0.000--0.010 |    4 | 0.009 | 0.000 | -0.2 |
| 0.010--0.025 |  712 | 0.020 | 0.010 | -2.7 |
| 0.025--0.050 | 3525 | 0.039 | 0.035 | -1.3 |
| 0.050--0.100 | 8272 | 0.075 | 0.073 | -0.7 |
| 0.100--0.150 | 6379 | 0.123 | 0.130 |  1.7 |
| 0.150--0.200 | 3860 | 0.172 | 0.175 |  0.5 |
| 0.200--0.250 | 2075 | 0.222 | 0.228 |  0.7 |
| 0.250--0.300 |  921 | 0.271 | 0.268 | -0.2 |
| 0.300--0.400 |  582 | 0.337 | 0.299 | -2.0 |
|       >0.400 |  151 | 0.480 | 0.450 | -0.7 |

<p class="text-sm text-zinc-500 dark:text-zinc-400"># races = 3198, # horses = 26481</p>
</figure>


The better fit provided by this model can be readily seen from the much smaller discrepancies between expected and actual frequencies. The parameter values used here should not be considered to be universal constants, as other authors have derived significantly different values for the parameters y and 6 using data from different racetracks (Lo, Bacon-Shone and Busche, 1994).

> To obtain the probabilities of subsequent places (second, third, etc.), we multiply the logits with a set of adjustment factors, which are themselves free parameters. For the loss function, we apply the Harville formula to compute the predicted probability for the actual finishing order.


```python
def softmax(logits: Tensor, adjustments: Tensor, places: int = 4) -> Tensor:
    """
    Compute the softmax probabilities with adjustments, following Benter's approach.

    Parameters:
    - logits (Tensor): A tensor containing the base utilities for each horse in each race.
    - adjustments (Tensor): A tensor containing the adjustments for each horse in each place.
    - places (int): The number of places to be considered.

    Returns:
    - Tensor: A tensor containing adjusted probabilities for each horse in each race for each place.

    Note:
    The function performs clamping to avoid NaNs when taking logarithms later.
    """

    # Compute the initial softmax probabilities based on logits
    p1 = F.softmax(logits, dim=1).unsqueeze(2)

    # Create a tensor by repeating p1 to match the number of places considered (excluding the first place)
    ps = p1.repeat(1, 1, places - 1)

    # Apply the Benter adjustments and recompute the softmax probabilities
    ps = F.softmax(adjustments * torch.log(torch.clamp(ps, min=1e-16)), dim=1)

    # Concatenate the initial probabilities with the adjusted probabilities for other places
    probs = torch.cat([p1, ps], dim=2)

    # Clamp the probabilities to avoid NaNs when applying the logarithm later
    probs = torch.clamp(probs, min=1e-16)

    return probs  # shape: races x horses x places
```


```python
def harville_loss(probs: Tensor, targets: Tensor, places: int = 4) -> Tensor:
    """
    Compute the Harville loss for a horse race betting model.

    Parameters:
    - probs (Tensor): A tensor containing the predicted probabilities for each horse in each race.
    - targets (Tensor): A tensor containing the true finishing positions for each horse in each race.
    - places (int): The number of places to be considered for the loss computation. Default is 4.

    Returns:
    - Tensor: A tensor containing the Harville loss, normalised by the number of places.

    Note: The function uses noise to handle dead heat places and performs clamping to avoid NaNs.
    """

    # Generate random noise to shuffle positions in case of dead heat
    noise = torch.rand(targets.shape).to(targets.device)

    # Sort targets based on noise-added values to handle dead heats
    targets = torch.argsort(targets + noise)

    # Compute the logarithm of the predicted probabilities
    log_probs = torch.log(probs)

    # Compute the initial negative log likelihood loss
    loss = F.nll_loss(log_probs[:, :, :places], targets[:, :places], reduction="none")
    loss = loss.sum(dim=1)

    # Adjust the loss for subsequent places
    for i in range(1, places):
        
        # Compute the probability for finishing in "i-th" place
        probs_place_i = -F.nll_loss(
            probs[:, :, i : i + 1].repeat(1, 1, i), targets[:, :i], reduction="none"
        )

        # Compute the denominator term for "i-th" place
        denominator_place_i = -torch.log(
            torch.clamp(1 - probs_place_i.sum(dim=1), min=1e-16)
        )

        # Adjust the loss
        loss -= denominator_place_i

    # Return the mean loss, normalised by the number of places
    return loss.mean() / places
```


```python
class PublicModel(pl.LightningModule):
    def __init__(self, places: int = 4, lr: float = 1e-3):
        super().__init__()
        self.example_input_array = torch.rand(32, 1)
        self.save_hyperparameters()
        self.adjustments = nn.Parameter(torch.ones(places - 1))

    def forward(self, p):
        public_log_probs = torch.log(p)
        return softmax(public_log_probs, self.adjustments, self.hparams.places)

    def training_step(self, batch, batch_idx):
        loss, _ = self._shared_evaluation(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_evaluation(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        return metrics

    def _shared_evaluation(self, batch, batch_idx):
        p, y = batch
        y_hat = self(p)
        loss = harville_loss(y_hat, y, self.hparams.places)
        acc = multiclass_accuracy(y_hat[:,:,0].argmax(dim=-1), y.argmin(dim=-1), 14)
        return loss, acc

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
```


```python
from lightning.pytorch.callbacks import LearningRateFinder
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

adjustments = []
for start, end in DATE_RANGES:
    val = (pd.Timestamp(end) + pd.DateOffset(years=1)).strftime('%Y-%m-%d')
    
    datamodule = pl.LightningDataModule.from_datasets(
        train_dataset=dataset(DATA_FILE, start, end),
        val_dataset=dataset(DATA_FILE, end, val),
        batch_size=32,
        num_workers=4,
    )

    model = PublicModel()

    early_stopping = EarlyStopping(monitor="val_loss", patience=6)
    learning_rate_finder = LearningRateFinder()
    trainer = pl.Trainer(
        callbacks=[early_stopping, learning_rate_finder],
        max_epochs=100,
        accelerator="gpu",
        devices=1,
    )
    trainer.fit(model=model, datamodule=datamodule)
    trainer.validate(model=model, datamodule=datamodule)
    
    γ, δ, ε = model.adjustments.tolist()
    adjustments.append((start, end, γ, δ, ε))
```

```python
for start, end, γ, δ, ε in adjustments:
    print(f"Public Estimate Adjustments ({start[:4]}-{end[:4]} Seasons)\nγ={γ:.2f}, δ={δ:.2f}, ε={ε:.2f}\n")
```

    Public Estimate Adjustments (1986-1993 Seasons)
    γ=0.83, δ=0.68, ε=0.58

    Public Estimate Adjustments (1996-2003 Seasons)
    γ=0.81, δ=0.75, ε=0.67

    Public Estimate Adjustments (2006-2013 Seasons)
    γ=0.88, δ=0.73, ε=0.67

    Public Estimate Adjustments (2016-2023 Seasons)
    γ=0.86, δ=0.74, ε=0.64



## Feasibility

A computer based handicapping and betting system could in principle be. developed and implemented at most of the world's racetracks. Today's portable computers have sufficient capacity not only for real-time calculation of the bets, but for model development as well. However, several important factors should be considered in selecting a target venue, as potential profitability varies <ins>considerably</ins> among racetracks. The following are a few practical recommendations based on the author's experience.

### Data availability

A reliable source of historical data must be available for developing the model and test samples. The track must have been in existence long enough, running races under conditions similar to today, in order to develop reliable predictions. Data availability in computer form is of great help, as data entry and checking are extremely time-consuming. The same data used in model development must also be available for real-time computer entry sufficient time before the start of each race. Additionally, final betting odds must be available over the development sample for the "combined model" estimation of equation (1) as well as for wagering simulations.

### Ease of operation

Having an accurate estimate of the final odds is imperative for betting purposes. Profitability will suffer greatly if the final odds are much different than the ones used to calculate the probabilities and bet sizes. The ideal venue is one which allows off-track telephone betting, and disseminates the odds electronically. This enables the handicapper to bet from the convenience of an office, and eliminates the need to take a portable computer to the track and type in the odds from the tote board at the last minute. Even given ideal circumstances, a professional effort will require several participants. Data entry and verification, general systems programming, and ongoing model development all require full-time efforts, as well as the day-today tasks of running a small business. Startup capital requirements are large, (mainly for research and development) unless the participants forgo salaries during the development phase.

### Beatability of the opposition

Pari-mutuel wagering is a competition amongst participants in a highly negative sum game. Whether a sufficiently effective model can be developed depends on the predictability of the racing, and the level of skill of fellow bettors. If the races are largely dishonest, and the public odds are dominated by inside information then it is unlikely that a fundamental model will perform well. Even if the racing is honest, if the general public skill level is high, or if some well financed minority is skillful, then the relative advantage obtainable will be less. Particularly unfavorable is the presence of other computer handicappers. Even independently developed computer models will probably have a high correlation with each other and thus will be lowering the dividends on the same horses, reducing the profitability for all. Unfortunately, it is difficult to know how great an edge can be achieved at a particular track until one develops a model for that track and tests it, which requires considerable effort. Should that prove successful, there is still no guarantee that the future will be as profitable as past simulations might indicate. The public may become more skillful, or the dishonesty of the races may increase, or another computer handicapper may start playing at the same time.

### Pool size limitations

Perhaps the most serious and inescapable limitation on profitability is a result of the finite amount of money in the betting pools. The high track take means that only the most extreme public probability mis-estimations will result in profitable betting opportunities, and the maximum bet size imposed by the bets' effects on the dividends limits the amount that can be wagered. Simulations by the author have indicated that a realistic estimate of the maximum expected profit achievable, as a percentage of total per-race turnover, is in the range of 0.25--0.5 per cent. This is for the case of a player with an effectively infinite bankroll. It may be true that at tracks with small pool sizes, that this percentage is higher due to the lack of sophistication of the public, but in any case, it is unlikely that this value could exceed 1.5 per cent. A more realistic goal for a start-up operation with a bankroll equal to approximately one half of the per-race turnover might be to win between 0.1 and 0.2 per cent of the total track turnover. The unfortunate implication of this is that at small volume tracks one could probably not make enough money for the operation to be viable.

Racetracks with small betting volumes also tend to have highly volatile betting odds. In order to have time to calculate and place one's wagers it is necessary to use the public odds available a few minutes before post time. The inaccuracy involved in using these volatile pre-post-time odds will decrease the effectiveness of the model.

## Results

The author has conducted a betting operation in Hong Kong following the principles outlined above for the past five years. Approximately five man-years of effort were necessary to organize the database and develop a handicapping model which showed a significant advantage. An additional five man-years were necessary to develop the operation to a high level of profitability. Under near-ideal circumstances, ongoing operations still require the full time effort of several persons.

A sample of approximately 2,000 races (with complete past performance records for each entrant) was initially used for model development and testing. Improvements to the model were made on a continuing basis, as were regular re-estimations of the model which incorporated the additional data accumulated. A conservative fractional Kelly betting strategy was employed throughout, with wagers being placed on all positive expectation bets available in both normal and exotic pools (except place and show bets). Extremely large pool sizes, (> USD $10,000,000 per race turnover) made for low volatility odds, therefore bets could be placed with accurate estimations of the final public odds. Bets were made on all available races except for races containing only *unratable* horses (-5%), resulting in approximately 470 races bet per year. The average track take was -19% during this period.

Four of the five seasons resulted in net profits, the loss incurred during the losing season being approximately 20% of starting capital. A strong upward trend in rate of return has been observed as improvements were made to the handicapping model. Returns in the various betting pools have correlated well with theory, with the rate-of-return in exotic pools being generally higher than that in simple pools. While a precise calculation has not been made, the statistical significance of this result is evident. Following is a graph of the natural logarithm of [(wealth) / (initial wealth)] versus races bet.

## Conclusion

The question; "Can a system beat the races?" can surely be answered in the affirmative. The author's experience has shown that at least at some times, at some tracks, a statistically derived fundamental handicapping model can achieve a significant positive expectation. It will always remain an empirical question whether the racing at a particular track at a particular time can be beaten with such a system. It is the author's conviction that we are now experiencing the *golden age* for such systems. Advances in computer technology have only recently made portable and affordable the processing power necessary to implement such a model. In the future, computer handicappers may become more numerous, or one of the racing publications may start publishing competent computer ratings of the horses, either of which will likely cause the market to become efficient to such predictions. The profits have gone, and will go, to those who are "in action" first with sophisticated models.

## References

[^benter08]: [Benter, W. (2008). Computer Based Horse Race Handicapping and Wagering Systems: A Report. *Efficiency of Racetrack Betting Markets*.](https://www.gwern.net/docs/statistics/decision/1994-benter.pdf)

[^benter96]: [Benter, W., Miel, G. J. & Turnbough P. D. (1996). Modelling Distance Preference in Thoroughbred Racehorses. *Third Conference on Mathematics and Computers in Sports*.](http://www.mathsportinternational.com/anziam/Mathsport%203.pdf)

[^chellel18]: [Chellel, K. (2018). The Gambler Who Cracked the Horse-Racing Code. *Bloomberg Businessweek.*](https://www.bloomberg.com/news/features/2018-05-03/the-gambler-who-cracked-the-horse-racing-code)

[^gu03]: Gu, M. G., Huang, C. Q. & Benter, W. (2003). Multinomial Probit Models for Competitive Horse Racing. *Working paper*.

[^wong11]: [Wong, C. X. (2011). Precision: Statistical and Mathematical Methods in Horse Racing. *Outskirts Press*.](https://www.amazon.com/Precision-Statistical-Mathematical-Methods-Racing/dp/1432768522)
