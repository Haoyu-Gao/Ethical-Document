---
tags:
- vision
- image-classification
datasets:
- Kaludi/food-category-classification-v2.0
widget:
- src: https://www.foodandwine.com/thmb/gv06VNqj1uUJHGlw5e7IULwUmr8=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/2012-r-xl-vegetable-sandwich-with-dill-sauce-2000-0984c1b513ae4af396aee039afa5e38c.jpg
  example_title: Bread
- src: https://cdn.britannica.com/34/176234-050-0E0C55C6/Glass-milk.jpg
  example_title: Dairy
- src: https://images-gmi-pmc.edge-generalmills.com/7c1096c7-bfd0-4806-a794-1d3001fe0063.jpg
  example_title: Dessert
- src: https://theheirloompantry.co/wp-content/uploads/2022/06/how-to-fry-eggs-perfectly-in-4-ways-the-heirloom-pantry.jpg
  example_title: Egg
- src: https://www.mashed.com/img/gallery/the-real-reason-fried-foods-are-so-popular-right-now/l-intro-1650327494.jpg
  example_title: Fried Food
- src: https://www.seriouseats.com/thmb/WzQz05gt5witRGeOYKTcTqfe1gs=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/butter-basted-pan-seared-steaks-recipe-hero-06-03b1131c58524be2bd6c9851a2fbdbc3.jpg
  example_title: Meat
- src: https://assets3.thrillist.com/v1/image/3097381/1200x600/scale;
  example_title: Seafood
- src: https://i0.wp.com/post.healthline.com/wp-content/uploads/2020/03/romaine-lettuce-1296x728-body.jpg?w=1155&h=1528
  example_title: Vegetable
co2_eq_emissions:
  emissions: 12.456278925446485
---

# Food Category Classification v2.0

This is an updated Food Category Image Classifier model of the [old](https://huggingface.co/Kaludi/food-category-classification) model that has been trained by [Kaludi](https://huggingface.co/Kaludi) to recognize **12** different categories of foods, which includes **Bread**, **Dairy**, **Dessert**, **Egg**, **Fried Food**, **Fruit**, **Meat**, **Noodles**, **Rice**, **Seafood**, **Soup**, and **Vegetable**. It can accurately classify an image of food into one of these categories by analyzing its visual features. This model can be used by food bloggers, restaurants, and recipe websites to quickly categorize and sort their food images, making it easier to manage their content and provide a better user experience.

### Gradio

This model supports a [Gradio](https://github.com/gradio-app/gradio) Web UI to run the data-food-classification model:
[![Open In HF Spaces](https://camo.githubusercontent.com/00380c35e60d6b04be65d3d94a58332be5cc93779f630bcdfc18ab9a3a7d3388/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f25463025394625413425393725323048756767696e67253230466163652d5370616365732d626c7565)](https://huggingface.co/spaces/Kaludi/Food-Category-Classification_V2_App)


## Validation Metrics
- Problem type: Multi-class Classification
- Model ID: 3353292434
- CO2 Emissions (in grams): 12.4563
- Loss: 0.144
- Accuracy: 0.960
- Macro F1: 0.959
- Micro F1: 0.960
- Weighted F1: 0.959
- Macro Precision: 0.962
- Micro Precision: 0.960
- Weighted Precision: 0.962
- Macro Recall: 0.960
- Micro Recall: 0.960
- Weighted Recall: 0.960