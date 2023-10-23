package com.example.lowscan

import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.ImageView
import android.widget.SeekBar
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.SwitchCompat
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.imgproc.Moments
import com.example.lowscan.orderPoints

class MainActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var seekbarBrightness: SeekBar
    private lateinit var switchFilter: SwitchCompat

    private var imageBitmap: Bitmap? = null
    private var imageMat: Mat? = null // Full resolution image
    private var processMat: Mat? = null // Reduced image for processing and temporary display
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        seekbarBrightness = findViewById(R.id.seekbarBrightness)
        switchFilter = findViewById(R.id.switchFilter)

        // Initialize OpenCV
        OpenCVLoader.initDebug()

        switchFilter.setOnCheckedChangeListener { compoundButton, b ->
            updateImage()
        }

        seekbarBrightness.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                updateImage()
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) {
                // empty
            }

            override fun onStopTrackingTouch(seekBar: SeekBar?) {
                // empty
            }
        })

        seekbarBrightness.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                updateImage()
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) {
                // empty
            }

            override fun onStopTrackingTouch(seekBar: SeekBar?) {
                // empty
            }
        })
    }

    private fun updateImage() {
        // Calculate brightness based on SeekBar progress
        val brightness = mapSeekBarProgressToValue(seekbarBrightness.progress, -100.0, 0.0)

        // Apply brightness to the image
        processImage(brightness, switchFilter.isChecked)
    }

    private fun mapSeekBarProgressToValue(
        progress: Int,
        minValue: Double,
        maxValue: Double
    ): Double {
        val range = maxValue - minValue
        return minValue + (progress / 100.0) * range
    }

    // Button click handler to capture or select an image
    fun onCaptureOrSelectImageClick(view: View) {
        imagePickerResult.launch("image/*")
    }

    private val imagePickerResult =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
            uri?.let {
                imageBitmap = MediaStore.Images.Media.getBitmap(contentResolver, uri)
                processImage()
            }
        }

    // Display the image
    private fun showImage() {
        val bwBitmap = Bitmap.createBitmap(processMat.cols(), processMat.rows(), Bitmap.Config.RGB_565)
        // Step 7: Convert the filtered grayscale Mat back to a Bitmap format
        Utils.matToBitmap(processMat, bwBitmap)
        // Step 8: Set the filtered Bitmap (bwBitmap) to be displayed in the ImageView
        imageView.setImageBitmap(bwBitmap)
    }

    // Process the captured or selected image with OpenCV and display the result
    private fun processImage(
        brightness: Double = -30.0,
        isGrayScale: Boolean = false
    ) {
        // Step 1: Create a Mat object to hold the image data
        imageMat = Mat()
        // Step 2: Convert the input imageBitmap to a Mat object (OpenCV format)
        Utils.bitmapToMat(imageBitmap, imageMat)
        // Scaling down the image for faster processing
        val origHeight = imageMat.rows()
        val origWidth = imageMat.cols()
        val tinyHeight = 500
        val scaleFactor = tinyHeight.toDouble() / origHeight
        val tinyWidth = (origWidth.toDouble() * scaleFactor).toInt()
        val tinySize = Size(tinyWidth.toDouble(),tinyHeight.toDouble())
//        val tinyMat = Mat()
        var grayMat = Mat()
        Imgproc.resize(imageMat,processMat,tinySize)

        // Extract channel with the best contrast for the ink to be detected
        val prefChannel = "Green"
        when (prefChannel) {
            "Red" -> {
                Core.extractChannel(processMat, processMat, 2) // Extract the Red channel (0-based index)
            }
            "Green" -> {
                Core.extractChannel(processMat, processMat, 1) // Extract the Green channel (0-based index)
            }
            "Blue" -> {
                Core.extractChannel(processMat, processMat, 0) // Extract the Blue channel (0-based index)
            }
            else -> {
                // Handle the case when 'prefChannel' is not one of the specified colors-> grayscale
                Imgproc.cvtColor(processMat,processMat,Imgproc.COLOR_BGR2GRAY)
            }
        }

        // Blur
        Imgproc.GaussianBlur(processMat,processMat, Size(5.0, 5.0),0.0)

        // Canny
//        var edgeMat = Mat()
        Imgproc.Canny(processMat,processMat,75.0,200.0)

        // -> Show Canny for debug? maybe user can close the quadrilateral themself
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(processMat.clone(), contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)

//        contours.sortWith(Comparator { c1, c2 ->
//            (Imgproc.contourArea(c1) - Imgproc.contourArea(c2)).toInt()
//        })

        // Sort the contours by area in descending order and take the top 5
        contours.sortByDescending { Imgproc.contourArea(it) }
        val selectedContours = contours.take(5)

        showImage()
//        # Loop over the contours
        var previousArea = -1.0
        val imageArea = tinyWidth * tinyHeight
        var finalCnt = MatOfPoint2f()
        var first = true

        for (c in contours) {
            // Convert the contour to MatOfPoint2f
            val c2f = MatOfPoint2f()
            c.convertTo(c2f, CvType.CV_32F)

            // Approximate the contour
            val peri = Imgproc.arcLength(c2f, true)
            val approx = MatOfPoint2f()
            Imgproc.approxPolyDP(c2f, approx, 0.02 * peri, true)

            // If our approximated contour has four points, then we
            // can assume that we have found our screen
            if (approx.rows() == 4) {
                val area = Imgproc.contourArea(approx)

                if (first && area < 0.99 * imageArea) {
                    // Eliminate too big rectangle (image border)
                    finalCnt = approx
                    previousArea = area
                } else if (0.7 * previousArea < area && area < 0.9 * previousArea) {
                    // We keep the last nested rectangle which is at least 70% as big as
                    // the previous one, but at most 85% as big as the previous
                    // to get the biggest one if two consecutive rectangles are very close
                    finalCnt = approx
                    previousArea = area
                }

                first = false
            }
        }

        val pointsOrdered= orderPoints(finalCnt)


//        # approximate the contour
//        peri = cv2.arcLength(c, True)
//        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
//        # if our approximated contour has four points, then we
//        # can assume that we have found our screen
//                if len(approx) == 4:
//        area = cv2.contourArea(approx)
//        # print(area)
//        if area > 0.7*previous_area and area <0.99*image_area:
//        # Keep last big nested rectangle and
//        # eliminate too big rectangle (image border)
//        # we take the last nested rectangle which is at least 70% as big as
//        # the previous one
//        # print("ok")
//        finalCnt = approx[:,0,:]
//        previous_area = area
//        screenCnt = approx[:,0,:]
//        if first:
//        # drawCnt(screenCnt,"g")
//        drawCnt(c[:,0,:],"g")
//        # print("first")
//        # print(screenCnt)
//        else:
//        # drawCnt(screenCnt,"b")
//        drawCnt(c[:,0,:],"b")
//        # print("not first")
//        # print(screenCnt)
//        first = False
//        else:
//        # drawCnt(approx[:,0,:],"r")
//        drawCnt(c[:,0,:],"r")
//        # print("not 4")
//        drawCnt(finalCnt,"m")
//
//        # Apply the four point transform to obtain a top-down
//        # view of the original image
//        # warped = four_point_transform(orig, finalCnt * ratio)
//        # Corrected ratio four point perspective transform
//                pts = orderPoints(finalCnt*ratio)
//        (pts2,W,H) = getPtsPerspective(pts, height_orig, width_orig)
//        M = cv2.getPerspectiveTransform(pts,pts2)
//        warped = cv2.warpPerspective(orig,M,(W,H))
//
//        # convert the warped image to grayscale, then threshold it
//        # to give it that 'black and white' paper effect
//        warped = warped[:,:,1]
//        warped = 255 - warped # Invert colors
//        # warped = imutils.resize(warped, height = 500)
//        plt.figure()
//        plt.imshow(warped,cmap="gray",vmin=0,vmax=255)
//        plt.show()
//        # warped_thresh = cv2.adaptiveThreshold(warped,255,cv2.ADAPTIVE_THRESH_MEAN_C,
//            #                                       cv2.THRESH_BINARY,101,5)
//        T = threshold_local(warped, 101, offset = 5, method = "gaussian")
//        warped_thresh = (warped > T).astype("uint8") * 255
//        # (otsu,warped_thresh) = cv2.threshold(warped, 0, 255,
//        #     cv2.THRESH_BINARY | cv2.THRESH_OTSU)
//
//        # Erode dilate operation to remove noise
//                kernel = np.ones((2, 2), np.uint8)
//        img_dilation = cv2.dilate(warped_thresh, kernel, iterations=1)
//        img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
        // */
//        Core.add(grayMat, Scalar.all(brightness), grayMat) // Adjust brightness (optional)
//        // Step 5: Adjust the brightness of the grayscale image (optional)
//        Core.add(grayMat, Scalar.all(brightness), grayMat) // Adjust brightness (optional)
    }
}