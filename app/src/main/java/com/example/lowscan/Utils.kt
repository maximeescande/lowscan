package com.example.lowscan

import org.opencv.core.Point
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
fun orderPoints(pts: MatOfPoint2f): MatOfPoint2f {
    // Sort the points based on their x-coordinates
    val xSorted = pts.toList().sortedBy { it.x }

    // Grab the left-most and right-most points from the sorted
    var leftMost = xSorted.subList(0, 2)
    var rightMost = xSorted.subList(2, 4)

    // Sort the left-most coordinates according to their y-coordinates
    leftMost = leftMost.toList().sortedBy { it.y }
    val tl = leftMost[0]
    val bl = leftMost[1]

    // Calculate the Euclidean distance between the top-left and right-most points
    val d = xSorted[0].dot(rightMost[0] - tl)
    val br = if (d < xSorted[1].dot(rightMost[0] - tl)) xSorted[1] else xSorted[0]
    val tr = if (d < xSorted[1].dot(rightMost[0] - tl)) xSorted[0] else xSorted[1]

    // Return the coordinates in top-left, top-right, bottom-left, and bottom-right order (zig-zag)
    val pZigzag = MatOfPoint2f(tl, tr, bl, br)
    return pZigzag
}