Dataset description:
The Kepler space telescope is a now-retired telescope orbiting Earth. It was launched with a mission of discovering Earth-sized planets orbiting other stars - in other words, exoplanets. The way this was done was by monitoring stars for periodic small drops in intensity, which would signify something passing in front of them.
The dataset contains every candidate that Kepler spotted over its tenure, as well as whether or not it was confirmed to be an exoplanet. 

---
Program description:

This program uses pipeline to cross-validate (including both randomized search and exhaustive search) several ML models given different pre-ML choices (PCA or non-PCA).
The program determines a final optimal model that can predict whether a planet is an exoplanet or not. 
Output includes a classification report and a confusion matrix.
