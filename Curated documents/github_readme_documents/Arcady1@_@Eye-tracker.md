<h1 align="center">
  <a href="https://scrolling-web-page-with-your-eyes.glitch.me/">
    <img src="https://github.com/Arcady1/Eye-tracker/blob/main/web/github/eye-tracker-logo.jpg" alt="" width="500px"></img>
  </a>
  <br>
  <a href="https://scrolling-web-page-with-your-eyes.glitch.me/">Eye-tracker</a>
  <br>
</h1>

This application allows you to scroll a web page with your head movements.

### How eye tracker works
1) The user's webcam detects his face and eyes 
2) The algorithm detects user blinks
3) After a double blink, the algorithm tracks the user's head movements to get the direction to scroll
4) Then the scroll starts and stops when the user blinks twice

![GIF][0]

### How it works
* Open this [DEMO][1]
* Allow access to the camera
* Wait for the algorithm to recognize your face 
* Make a double blink to start scroll (the "unlock" icon will appear in the upper right corner)
* Nod up or down
* Make a double blink to stop scroll (the "lock" icon will appear in the upper right corner)

### How to use

To clone and run this application, you'll need Git. From your command line:

```
# Clone this repository
$ git clone https://github.com/Arcady1/Eye-tracker.git

# Go into the repository
$ cd Eye-tracker

# Install dependencies
$ npm install

# Open the index.html file
```

### Credits
This software uses the following open source packages:

* [jQuery][2] "^3.5.1"
* [Browserify][3] "latest"
* [TensorFlow][4] "2.6.0"
* [Face landmarks detection model][4.2] "0.0.1"
* [Chart.js][4.3] "^2.9.4"

### Acknowledgments
* [TensorFlow Blog][5]
* [MediaPipe][4.1]
* [MediaPipe Iris][5.2]
* [MediaPipe Iris MODEL CARD][5.3]

### You may also like...
* [Don't touch your face][6] - Face touch detection with BodyPix segmentation
* [Doodle Recognition][7] - Web app classsificator based on the Quick, Draw! Dataset.
* [Pomodoro Bot][8] - Telegram bot with the pomodoro timer

### License
MIT

[0]: https://github.com/Arcady1/Scrolling-web-page-with-your-eyes/blob/main/web/github/eye-control-gif.gif

[1]: https://scrolling-web-page-with-your-eyes.glitch.me/
[2]: https://github.com/jquery/jquery
[3]: https://github.com/browserify/browserify
[4]: https://github.com/tensorflow/tfjs
[4.1]: https://github.com/google/mediapipe
[4.2]: https://blog.tensorflow.org/2020/11/iris-landmark-tracking-in-browser-with-MediaPipe-and-TensorFlowJS.html
[4.3]: https://github.com/chartjs/Chart.js

[5]: https://blog.tensorflow.org/search?label=TensorFlow.js&max-results=20
[5.2]: https://google.github.io/mediapipe/solutions/iris
[5.3]: https://drive.google.com/file/d/1bsWbokp9AklH2ANjCfmjqEzzxO1CNbMu/view

[6]: https://github.com/Arcady1/Do-not-touch-your-face
[7]: https://github.com/Arcady1/Doodle-Recognition-Web
[8]: https://github.com/Arcady1/Telegram-Pomodoro-Bot
