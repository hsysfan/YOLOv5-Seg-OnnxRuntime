using OnnxSample.Yolov5;
using OpenCvSharp;
using System;

namespace OnnxSample
{
    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length == 0)
            {
                Console.WriteLine("Usage: OnnxSample.exe [mode] (det or seg)");
                return;
            }
            if (args[0] == "det")
            {
                var detector = new YoloDetector("sample-model.onnx");
                using (var image = Cv2.ImRead("simple_test.jpg"))
                {
                    float ratio = 0.0f;
                    Point diff1 = new Point();
                    Point diff2 = new Point();
                    var letter_image = YoloDetector.CreateLetterbox(image, new Size(640, 384), new Scalar(114, 114, 114), out ratio, out diff1, out diff2);
                    var result = detector.objectDetection(image);

                    Cv2.NamedWindow("SOURCE", WindowMode.Normal);
                    Cv2.ImShow("SOURCE", image);
                    Cv2.NamedWindow("LETTERBOX", WindowMode.Normal);
                    Cv2.ImShow("LETTERBOX", letter_image);
                    using (var dispImage = image.Clone())
                    {
                        foreach (var obj in result)
                        {
                            Cv2.Rectangle(dispImage, new Point(obj.Box.Xmin, obj.Box.Ymin), new Point(obj.Box.Xmax, obj.Box.Ymax), new Scalar(0, 0, 255), 2);
                            Cv2.PutText(dispImage, obj.Label, new Point(obj.Box.Xmin, obj.Box.Ymin - 5), HersheyFonts.HersheySimplex, 1, new Scalar(0, 0, 255), 2);
                        }
                        Cv2.NamedWindow("RESULT", WindowMode.Normal);
                        Cv2.ImShow("RESULT", dispImage);
                    }
                    Cv2.WaitKey();
                }
            }
            else if (args[0] == "seg")
            {
                var dataSize = 640;
                var detector = new YoloDetector("sample-seg-model.onnx")
                {
                    imgSize = new Size(dataSize, dataSize),
                    MinConfidence = 0.7f
                };
                using (var image = Cv2.ImRead("simple_seg_test.jpg"))
                {
                    var result = detector.objectSegmentation(image);
                    var result_color = new Mat();
                    result_color = result.CvtColor(ColorConversionCodes.GRAY2BGR);

                    var dispImage = new Mat(new Size(2 * dataSize, dataSize), MatType.CV_8UC3);

                    Cv2.HConcat(image, result_color, dispImage);
                    Cv2.ImShow("result", dispImage);
                    Cv2.WaitKey();
                }
            }
            else
            {
                Console.WriteLine("Usage: OnnxSample.exe [mode] (det or seg)");
                return;
            }
        }
    }
}
