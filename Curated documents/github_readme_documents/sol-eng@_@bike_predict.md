# Bike Predict End-to-End Machine Learning Pipeline

This repository contains an example of using [Posit Connect](https://rstudio.com/products/connect/), [pins](https://github.com/rstudio/pins), and [vetiver](https://vetiver.tidymodels.org) to create an end-to-end machine learning pipeline.

![](img/arrows.drawio.png)

## Who This is For

Both *data scientists* and *R admins* in machine-learning-heavy contexts may find this demo interesting. People who describe *production-izing* or *deploying* content as pain points may find this helpful.

Some particular pain points this could address:

### I am trying to deploy/productionize a machine learning model

People mean MANY different things by "productionize" a machine learning model. Very often, that means making the output of a model available to another process. The most common paths to making model output accessible to other tools are writing to a database, writing to a flat file (or pin), or providing real-time predictions with a plumber API.

This repository contains examples of all three of these patterns. The model metrics script outputs the test data, including predictions, to a database and outputs model performance metrics to a pin. It would be easy to make either of these the final consumable for another process, like the shiny client app. The shiny app in this repository uses a plumber API serving predictions.

Another common problem in deploying the model is figuring out where the model lives. In this example, the model(s) is(are) pinned to Posit Connect and are consumed by the other assets (test data script and plumber API) from the pin.

For relatively advanced model deployments, users may be interested in horseracing different models, A/B testing one model from another, or monitoring model performance and drift over time. Once finished, the model performance dashboard will be a tool to compare different models and examine model performance over time.

Another piece embedded in the background of deploying/productionizing a model is making the entire pipeline robust to (for example) someone accidentally pushing the deploy button when they shouldn't. A perfect solution to this is programmatic deployment. This entire repository is deployed from a GitHub repo, using the functionality in Posit Connect. One cool problem this can solve is deploying dev versions of content, which can easily be accomplished using a long-running deployed dev branch. There's an example of this in the Dev Client App.

Another piece of this is making the underlying R functions more robust. See the next point for more on that.

### I have a bunch of functions I need to use, but it's a pain

Most R users know that the correct solution is to put their R functions into a package if they are reused -- or even if they need to be well-documented and tested. This repository includes a package of helper functions that do various tasks.

Many R users aren't sure how to deploy their package. Their packages work well locally, but everything breaks when they try to deploy. This is a great use case for Posit Package Manager. Posit Package Manager makes it easy to create a package that contains the code needed in an app, push that code up to git, and have it available via `install.packages` to a deployment environment (like Posit Connect) that might need it.

For more details, see the [Posit Package Manager](https://rstudio.com/products/package-manager/) page and <https://environments.rstudio.com>.

### I have a bunch of CSV files I use in my shiny app

For some workflows, a CSV file is the best choice for storing data. However, for many (most?) cases, the data would do better if stored somewhere centrally accessible by multiple people where the latest version is always available. This is particularly true if that data is reused across multiple projects or pieces of content.

This project has two data files that are particularly pin-able -- the station metadata file (that maps station IDs to names and locations) and the data frame of out-of-sample error metrics for each model. Both are relatively small files, reused by multiple assets, where only the newest version is needed -- perfect candidates for a pin.

A few other non-dataset objects are also perfect for a pin: the models themselves and the test/training split. These have similar properties to the datasets -- small, reused, and only the newest is needed -- and are serializable by R, making them excellent choices for a pin.

Some examples of objects that are likely to be a good fit for a pin:

-   machine-learning models
-   plotting data that is updated on a schedule (as opposed to created on demand)
-   data splits/training data sets
-   metadata files from some machine-readable ID to human-readable details

### I've got this CRON job that does some ETL/data processing/creates a bunch of files

Scheduled R Markdown isn't *always* the best solution here (for example, robust SQL pipelines in another tool don't need to be replaced with scheduled RMarkdown), but if the user is running R code, scheduled R Markdown is way easier than anything else.

## What doesn't it do

This repository shows an exciting set of capabilities, combining open-source R and Python with Posit's professional products. There are a few things it doesn't do (yet) -- but that I might add, depending on interest:

-   Jobs don't depend on another. I've scheduled the jobs so that each will complete by the time another starts, but there are tools in R (like [drake](https://github.com/ropensci/drake)) that allow you to put the entire pipeline into code and make dependencies explicit.
-   Pieces of content must be managed individually, including uploading, permissions, environment variables, and tagging. It is possible to do something more robust via programmatic deployment using the Posit Connect API, but generic git deployment doesn't support deploying all of the content in a git repo at once.

## Individual Content

| Content                                   | Description                                                  | Code                                                         | Content Deployed to Connect                                  |
| ----------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **ETL** Step 01 - Raw Data Refresh        | Get the latest station status data from the <https://capitalbikeshare.com> API. The data is written to *Content DB* in table *bike_raw*\_data and *bike_station_info*. | [content/01-etl/01-raw-data-refresh/document.qmd](content/01-etl/01-raw-data-refresh/document.qmd) | [Quarto document](https://colorado.rstudio.com/rsc/bike-predict-r-raw-data-refresh/), [Pin (bike_station_info)](https://colorado.rstudio.com/rsc/bike-predict-r-station-info-data-pin/) |
| **ETL** Step 2 - Tidy data                | From *Content DB* get two tables: (1) *bike_raw*\_data and (2) *bike_station_info*. The two data sets are tidied and then combined. The resulting tidy data set is written to *Content DB* in table *bike_model_data*. | [content/01-etl/02-tidy-data/document.qmd](content/01-etl/02-tidy-data/document.qmd) | [Quarto document](https://colorado.rstudio.com/rsc/bike-predict-r-tidy-data/) |
| **Model** Step 1 - Train and Deploy Model | From *Content DB* get the *bike_model_data* table and then train a model. The model is saved to Connect as a pin, and then deployed to Connect as a plumber API using vetiver. | [content/02-model/01-train-and-deploy-model/document.qmd](content/02-model/01-train-and-deploy-model/document.qmd) | [Quarto document](https://colorado.rstudio.com/rsc/bike-predict-r-train-and-deploy-model/), [Pin](https://colorado.rstudio.com/rsc/bike-predict-r-pinned-model/), [Plumber API](https://colorado.rstudio.com/rsc/bike-predict-r-api/) |
| **Model** Step 2 - Model Card             | Use the [vetiver model card template](https://vetiver.rstudio.com/learn-more/model-card.html) to document essential facts and considerations of the deployed model. | [content/02-model/03-model-card/document.qmd](content/02-model/03-model-card/document.qmd) | [Quarto document](https://colorado.rstudio.com/rsc/bike-predict-r-model-card/) |
| **Model** Step 3 - Model Metrics          | Use vetiver to document the model performance. Model performance metrics are calculated and then written to pin using vetiver. | [content/02-model/02-model-metrics/document.qmd](content/02-model/02-model-metrics/document.qmd) | [Quarto document](https://colorado.rstudio.com/rsc/bike-predict-r-model-metrics/), [Pin](https://colorado.rstudio.com/rsc/bike-predict-r-model-metrics-pin/) |
| **App** - Client App                      | Use the API endpoint to interactively server predictions to a shiny app.| [content/03-app/01-client-app/app.R](content/03-app/01-client-app/app.R)                                           | [Shiny app](https://colorado.rstudio.com/rsc/bike-predict-r-client-app/)                                                                                                                                                              |
| **App** - Dev Client App                  | A development version of the client app.                                | [content/03-app/03-client-app-dev/app.R](content/03-app/03-client-app-dev/app.R)                                   | [Shiny app](https://colorado.rstudio.com/rsc/bike-predict-r-client-app-dev/)                                                                                                                                                              |
| **App** - Content Dashboard               | A dashboard that contains links to all of the bike predict content.     | [content/03-app/02-connect-widgets-app/document.qmd](content/03-app/02-connect-widgets-app/document.qmd)           | [Quarto document](https://colorado.rstudio.com/rsc/bike-predict-r-dashboard/)                                                                                                                                                         |

## Contributing

See a problem or want to contribute? Please refer to the [contributing page](./CONTRBUTING.md).
