# DS4420 Final Project
Tanishi Datta and Shruthi Palaniappan 

## Project Overview

Cooking often requires adapting recipes when certain ingredients are unavailable or when substitutions are needed for dietary restrictions or personal preferences. However, identifying appropriate substitutions can be challenging because ingredients not only affect the flavor of a dish but the texture and structure.

This project explores whether patterns in recipe ingredient combinations can be used to identify potential food substitutions. Using a large recipe dataset from Food.com, we analyze how ingredients commonly appear together across thousands of recipes. By identifying patterns of ingredient co-occurrence, we aim to discover ingredients that play similar roles in recipes and could potentially serve as substitutes.

The goal of this project is to apply machine learning techniques to understand ingredient relationships and evaluate whether these patterns can generate reasonable substitution suggestions.

## Dataset
The dataset is sourced from Kaggle and consists of 180K+ recipes and 700K+ recipe reviews pulled for Food.com. Each recipe includes information such as the time it takes to prepare, any tags from Food.com, nutrition information, number of steps as well as the text for each step, and most importantly a list of ingredients and total number of ingredients. 

## Methods 
To identify potential ingredient substitutions, we plan to implement two modeling approaches:

### Bayesian Model 
The Bayesian model estimates probabilistic relationships between ingredients.

### Collaborative Filtering 
Collaborative filtering leverages patterns of ingredient co-occurrence across recipes. By treating recipes like users and ingredients as items, this approach identifies ingredients that frequently appear in similar recipe contexts. 
