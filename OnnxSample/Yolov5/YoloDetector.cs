using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Linq;

namespace OnnxSample.Yolov5
{
    public class YoloDetector : IDisposable
    {
        private InferenceSession sess = null;
        private Mat imageFloat = null;
        private Mat debugImage = null;
        public float MinConfidence { get; set; }
        public float NmsThresh { get; set; }
        private float maxWH = 4096;
        public Size imgSize = new Size(640, 384);
        private Scalar padColor = new Scalar(114, 114, 114);

        /// <summary>
        /// Initialize
        /// </summary>
        /// <param name="model_path"></param>
        public YoloDetector(string model_path)
        {
            //var option = new SessionOptions();
            //option.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            //option.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;            

            //sess = new InferenceSession(model_path, option);

            //imageFloat = new Mat();
            //debugImage = new Mat();
            //MinConfidence = 0.2f;
            //NmsThresh = 0.4f;

            int gpuDeviceId = 0; // The GPU device ID to execute on
            sess = new InferenceSession(model_path, SessionOptions.MakeSessionOptionWithCudaProvider(gpuDeviceId));
            imageFloat = new Mat();
            debugImage = new Mat();
            MinConfidence = 0.25f;
            NmsThresh = 0.45f;
        }


        public List<Prediction> objectDetection(Mat img, double confidence = 0.4)
        {
            MinConfidence = (float)confidence;
            float ratio = 0.0f;
            Point diff1 = new Point();
            Point diff2 = new Point();
            List<Prediction> obj_list = new List<Prediction>();
            //Image -> Letterbox
            bool isAuto = true;
            if (img.Width <= imgSize.Width || img.Height <= imgSize.Height) isAuto = false;
            using (var letterimg = CreateLetterbox(img, imgSize, padColor, out ratio, out diff1, out diff2, auto: isAuto, scaleFill: !isAuto))
            {
                letterimg.ConvertTo(imageFloat, MatType.CV_32FC3, (float)(1 / 255.0));
                var input = new DenseTensor<float>(MatToList(imageFloat), new[] { 1, 3, imgSize.Height, imgSize.Width });
                // Setup inputs and outputs
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("images", input)
                };

                using (var results = sess.Run(inputs))
                {
                    //Postprocessing
                    var resultsArray = results.ToArray();
                    var pred_value = resultsArray[0].AsEnumerable<float>().ToArray();
                    var pred_dim = resultsArray[0].AsTensor<float>().Dimensions.ToArray();

                    var nc = pred_dim[pred_dim.Length - 1] - 5;
                    var candidate = GetCandidate(pred_value, pred_dim, MinConfidence);
                    //Compute conf
                    for (int i = 0; i < candidate.Count; i++)
                    {
                        var obj_cnf = candidate[i][4];
                        for (int j = 5; j < candidate[i].Count; j++)
                        {
                            candidate[i][j] *= obj_cnf;
                        }
                    }

                    //Change Box coord (xywh -> xyxy)
                    for (int i = 0; i < candidate.Count; i++)
                    {
                        var xmin = candidate[i][0] - candidate[i][2] / 2; //top left x
                        var ymin = candidate[i][1] - candidate[i][3] / 2; //top left y
                        var xmax = candidate[i][0] + candidate[i][2] / 2; //bottom right x
                        var ymax = candidate[i][1] + candidate[i][3] / 2; //bottom right y
                        candidate[i][0] = xmin;
                        candidate[i][1] = ymin;
                        candidate[i][2] = xmax;
                        candidate[i][3] = ymax;
                    }
                    //Detections matrix
                    var detected_mat = GetDetectionMatrix(candidate, MinConfidence);
                    //NMS
                    List<Rect> bboxes = new List<Rect>();
                    List<float> confidences = new List<float>();
                    for (int i = 0; i < detected_mat.Count; i++)
                    {
                        var diff_class = (int)(maxWH * detected_mat[i][5]);

                        Rect box = new Rect((int)detected_mat[i][0] + diff_class, (int)detected_mat[i][1] + diff_class,
                            (int)(detected_mat[i][2] - detected_mat[i][0]), (int)(detected_mat[i][3] - detected_mat[i][1]));
                        bboxes.Add(box);
                        confidences.Add(detected_mat[i][4]);
                    }
                    int[] indices = null;
                    CvDnn.NMSBoxes(bboxes, confidences, MinConfidence, NmsThresh, out indices);

                    var predictions = new List<Prediction>();
                    if (indices != null)
                    {
                        for (int ids = 0; ids < indices.Length; ids++)
                        {
                            int idx = indices[ids];
                            var cls = detected_mat[idx][detected_mat[idx].Count - 1];
                            var confi = detected_mat[idx][4];
                            predictions.Add(new Prediction
                            {
                                Box = new Box {
                                    Xmin = detected_mat[idx][0],
                                    Ymin = detected_mat[idx][1],
                                    Xmax = detected_mat[idx][2],
                                    Ymax = detected_mat[idx][3] },
                                Label = LabelMap.Labels[(int)cls],
                                Id = (int)cls,
                                Confidence = confi
                            });
                        }
                    }
                    //Rescale Predictions
                    var rescale_predictions = new List<Prediction>();
                    for (int ids = 0; ids < predictions.Count; ids++)
                    {
                        var pred = predictions[ids];
                        var rescaleBox = CalcRescaleBox(pred.Box, img.Size(), imgSize, diff1, diff2);
                        rescale_predictions.Add(new Prediction
                        {
                            Box = rescaleBox,
                            Label = pred.Label,
                            Id = pred.Id,
                            Confidence = pred.Confidence
                        });
                    }
                    return rescale_predictions;
                }
            }
        }

        public Mat objectSegmentation(Mat image)
        {
            float ratio = 0.0f;
            Point diff1 = new Point();
            Point diff2 = new Point();
            bool isAuto = true;
            if (image.Width <= imgSize.Width || image.Height <= imgSize.Height) isAuto = false;
            List<Mat> masks = new List<Mat>();

            List<string> result_List = new List<string>();
            using (var letterimg = CreateLetterbox(image, imgSize, padColor, out ratio, out diff1, out diff2, auto: isAuto, scaleFill: !isAuto))
            {
                letterimg.ConvertTo(imageFloat, MatType.CV_32FC3, (float)(1 / 255.0));
                int[] rdspan = new int[] { 1, 3, imgSize.Height, imgSize.Width };
                ReadOnlySpan<int> odimensions = new ReadOnlySpan<int>(rdspan);
                float[] denset = MatToList(imageFloat);
                var input = new DenseTensor<float>(denset, odimensions);
                // Setup inputs and outputs
                var inputs = new List<NamedOnnxValue>
                {
                    //NamedOnnxValue.CreateFromTensor("input", input)
                    NamedOnnxValue.CreateFromTensor("images", input)
                };
                using (var results = sess.Run(inputs))
                {

                    var resultsArray = results.ToArray();
                    var pred_value = resultsArray[0].AsEnumerable<float>().ToArray();
                    var pred_dim = resultsArray[0].AsTensor<float>().Dimensions.ToArray();

                    float[] mask_in = new float[pred_value.Length];

                    var pred_value_m = resultsArray[1].AsEnumerable<float>().ToArray();
                    var pred_dim_m = resultsArray[1].AsTensor<float>().Dimensions.ToArray();
                    int height = (int)pred_dim_m[2];
                    int width = (int)pred_dim_m[3];

                    int w_ratio = (int)((float)pred_dim_m[3] / (float)imgSize.Width);
                    int h_ratio = (int)((float)pred_dim_m[2] / (float)imgSize.Height);

                    var nc = pred_dim[pred_dim.Length - 1] - 5 - 32; //all - number of mask = nc
                    var candidate = GetCandidate(pred_value, pred_dim, MinConfidence); //xc

                    for (int i = 0; i < candidate.Count; i++)
                    {
                        var obj_cnf = candidate[i][4];
                        for (int j = 5; j < candidate[i].Count; j++)
                        {
                            candidate[i][j] *= obj_cnf;
                        }
                    }

                    //Change Box coord (xywh -> xyxy)
                    for (int i = 0; i < candidate.Count; i++)
                    {
                        var xmin = candidate[i][0] - candidate[i][2] / 2; //top left x
                        var ymin = candidate[i][1] - candidate[i][3] / 2; //top left y
                        var xmax = candidate[i][0] + candidate[i][2] / 2; //bottom right x
                        var ymax = candidate[i][1] + candidate[i][3] / 2; //bottom right y
                        candidate[i][0] = xmin;
                        candidate[i][1] = ymin;
                        candidate[i][2] = xmax;
                        candidate[i][3] = ymax;
                    }

                    var detected_mat = GetSegmentationMatrix(candidate, nc, MinConfidence);

                    List<Rect> bboxes = new List<Rect>();
                    List<float> confidences = new List<float>();
                    for (int i = 0; i < detected_mat.Count; i++)
                    {
                        var diff_class = (int)(maxWH * detected_mat[i][5]);

                        Rect box = new Rect((int)detected_mat[i][0] + diff_class, (int)detected_mat[i][1] + diff_class,
                            (int)(detected_mat[i][2] - detected_mat[i][0]), (int)(detected_mat[i][3] - detected_mat[i][1]));
                        bboxes.Add(box);
                        confidences.Add(detected_mat[i][4]);
                    }
                    int[] indices = null;

                    CvDnn.NMSBoxes(bboxes, confidences, MinConfidence, NmsThresh, out indices);

                    var predictions = new List<Prediction>();
                    if (indices != null)
                    {
                        for (int ids = 0; ids < indices.Length; ids++)
                        {
                            int idx = indices[ids];
                            var cls = detected_mat[idx][detected_mat[idx].Count - 1];
                            var confi = detected_mat[idx][4];
                            float xmin = detected_mat[idx][0] < 0 ? 0 : detected_mat[idx][0];
                            float ymin = detected_mat[idx][1] < 0 ? 0 : detected_mat[idx][1];
                            float xmax = detected_mat[idx][2] > imgSize.Width ? imgSize.Width : detected_mat[idx][2];
                            float ymax = detected_mat[idx][3] > imgSize.Height ? imgSize.Height : detected_mat[idx][3];

                            predictions.Add(new Prediction
                            {
                                Box = new Box
                                {
                                    Xmin = xmin,
                                    Ymin = ymin,
                                    Xmax = xmax,
                                    Ymax = ymax
                                },
                                //Label = ((int)cls).ToString(),
                                Label = LabelMap.Labels[(int)cls],
                                Id = (int)cls,
                                Confidence = confi
                            });
                        }
                    }

                    var rescale_predictions = new List<Prediction>();
                    for (int ids = 0; ids < predictions.Count; ids++)
                    {
                        var pred = predictions[ids];
                        var rescaleBox = CalcRescaleBox(pred.Box, image.Size(), imgSize, diff1, diff2);
                        rescale_predictions.Add(new Prediction
                        {
                            Box = rescaleBox,
                            Label = pred.Label,
                            Id = pred.Id,
                            Confidence = pred.Confidence
                        });
                    }

                    float[][] detection_list = candidate.Select(a => a.ToArray()).ToArray();

                    var proto = Makeproto(pred_value_m, pred_dim_m);

                    Mat save_result = new Mat(pred_dim_m[2], pred_dim_m[3], MatType.CV_32FC1, 0);

                    for (int index = 0; index < rescale_predictions.Count; index++)
                    {
                        for (int j = 0; j < 32; j++)
                        {
                            save_result += detection_list[index][j + 6] * proto[j];

                        }
                    }

                    for (int i = 0; i < pred_dim_m[2]; i++)
                    {
                        for (int j = 0; j < pred_dim_m[3]; j++)
                        {
                            if (Sigmoid(save_result.Get<float>(i, j)) > 0.5f)
                            {
                                save_result.Set(i, j, 1.0f);
                            }
                            else
                            {
                                save_result.Set(i, j, 0.0f);
                            }
                        }
                    }

                    Mat reshape_result = new Mat(rows: imgSize.Height, cols: imgSize.Width, type: MatType.CV_32FC1);

                    Cv2.Resize(save_result, reshape_result, imgSize);

                    Mat bg_mat = new Mat(imgSize, MatType.CV_32FC1, Scalar.All(0));

                    for (int i = 0; i < rescale_predictions.Count; i++)
                    {
                        for (int j = i + 1; j < rescale_predictions.Count; j++)
                        {
                            Rect rect1 = new Rect(new Point((int)rescale_predictions[i].Box.Xmin, (int)rescale_predictions[i].Box.Ymin), new Size(rescale_predictions[i].Box.Xmax - rescale_predictions[i].Box.Xmin, rescale_predictions[i].Box.Ymax - rescale_predictions[i].Box.Ymin));
                            Rect rect2 = new Rect(new Point((int)rescale_predictions[j].Box.Xmin, (int)rescale_predictions[j].Box.Ymin), new Size(rescale_predictions[j].Box.Xmax - rescale_predictions[j].Box.Xmin, rescale_predictions[j].Box.Ymax - rescale_predictions[j].Box.Ymin));
                            if (IntersectionOverUnion(rect1, rect2) >= 0.9)
                            {
                                rescale_predictions.Remove(rescale_predictions[j]);
                            }
                        }
                    }
                    foreach (Prediction a in rescale_predictions)
                    {
                        Mat seg_mat = new Mat(imgSize, MatType.CV_32FC1, Scalar.All(0));
                        int width_b = (int)Math.Abs(a.Box.Xmax - a.Box.Xmin);
                        int height_b = (int)Math.Abs(a.Box.Ymax - a.Box.Ymin);
                        Rect rect = new Rect(new Point((int)a.Box.Xmin, (int)a.Box.Ymin), new Size(width_b, height_b));
                        reshape_result[rect].CopyTo(seg_mat[rect]);
                        masks2segments(seg_mat, ref bg_mat);
                    }

                    return bg_mat;
                    
                }
            }
        }

        public void Dispose()
        {
            debugImage?.Dispose();
            debugImage = null;
            imageFloat?.Dispose();
            imageFloat = null;
            sess?.Dispose();
            sess = null;
        }
        public bool IntersectsWith(Rect rect1, Rect rect2)
        {
            return (rect1.X < rect2.X + rect2.Width) &&
                   (rect1.X + rect1.Width > rect2.X) &&
                   (rect1.Y < rect2.Y + rect2.Height) &&
                   (rect1.Y + rect1.Height > rect2.Y);
        }

        public Rect Intersect(Rect rect1, Rect rect2)
        {
            if (!rect1.IntersectsWith(rect2))
                return Rect.Empty;

            double x = Math.Max(rect1.X, rect2.X);
            double y = Math.Max(rect1.Y, rect2.Y);
            double width = Math.Min(rect1.X + rect1.Width, rect2.X + rect2.Width) - x;
            double height = Math.Min(rect1.Y + rect1.Height, rect2.Y + rect2.Height) - y;

            return new Rect((int)x, (int)y, (int)width, (int)height);
        }

        public double IntersectionOverUnion(Rect rect1, Rect rect2)
        {
            Rect intersection = rect1.Intersect(rect2);
            double intersectionArea = intersection.Width * intersection.Height;

            double unionArea = (rect1.Width * rect1.Height) + (rect2.Width * rect2.Height) - intersectionArea;

            return intersectionArea / unionArea;
        }
        public List<Mat> Makeproto(float[] proto, int[] shape)
        {
            List<Mat> masks = new List<Mat>();
            for (int batch = 0; batch < shape[0]; batch++)
            {
                for (int cls = 0; cls < shape[1]; cls++)
                {
                    float[] subData = new float[shape[2] * shape[3]];
                    for (int h = 0; h < shape[2]; h++)
                    {
                        for (int w = 0; w < shape[3]; w++)
                        {
                            int idx = (batch * shape[1] * shape[2] * shape[3]) +
                                (cls * shape[2] * shape[3]) + (h * shape[3]) + w;

                            subData[h * shape[2] + w] = proto[idx];
                        }
                    }
                    using (var mask = new Mat(new int[] { shape[2], shape[3] }, MatType.CV_32FC1, subData))
                    {
                        masks.Add(mask.Clone());
                    }
                }
            }

            return masks;
        }

        private float Sigmoid(float x)
        {
            return 1 / (1 + (float)Math.Exp(-x));
        }

        public void masks2segments(Mat masks, ref Mat bg_mask)
        {
            Mat src = new Mat();
            Mat red = new Mat();
            Mat inner = new Mat();
            bg_mask.ConvertTo(bg_mask, MatType.CV_8UC1);
            bg_mask.CopyTo(inner);
            masks.ConvertTo(src, MatType.CV_8UC1);
            bg_mask *= 255;
            src *= 255;

            string[] a = new string[1];
            Point[][] contours;
            HierarchyIndex[] hierarchy;
            Mat bin = new Mat();

            src.CopyTo(bin);
            Cv2.Threshold(bin, bin, 127, 255, ThresholdTypes.Binary);
            Cv2.FindContours(bin, out contours, out hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

            Cv2.BitwiseOr(inner, bin, bg_mask);
        }

        private Mat DataPreprocessing(Mat image)
        {
            Mat data = Mat.Zeros(image.Size(), MatType.CV_32FC3);
            using (var rgbImage = new Mat())
            {
                Cv2.CvtColor(image, rgbImage, ColorConversionCodes.BGR2RGB);
                rgbImage.ConvertTo(data, MatType.CV_32FC3, (float)(1 / 255.0));
                var channelData = Cv2.Split(data);
                channelData[0] = (channelData[0] - 0.485) / 0.229;
                channelData[1] = (channelData[1] - 0.456) / 0.224;
                channelData[2] = (channelData[2] - 0.406) / 0.225;
                Cv2.Merge(channelData, data);
            }
            return data;
        }

        private unsafe static float[] Create(float* ptr, int ih, int iw, int chn)
        {
            float[] array = new float[ih * iw * chn];

            for (int y = 0; y < ih; y++)
            {
                for (int x = 0; x < iw; x++)
                {
                    for (int c = 0; c < chn; c++)
                    {
                        var idx = (y * chn) * iw + (x * chn) + c;
                        var idx2 = (c * iw) * ih + (y * iw) + x;
                        array[idx2] = ptr[idx];
                    }
                }
            }
            return array;
        }

        public static Box CalcRescaleBox(Box dBox, Size orgImage, Size resizeImage, Point diff1, Point diff2)
        {
            Box rescaleBox = new Box {
                Xmin = 0,
                Ymin = 0,
                Xmax = 0,
                Ymax = 0
            };
            Point rImgStart = new Point(diff1.X + diff2.X, diff1.Y + diff2.Y);
            Point rImgEnd = new Point(resizeImage.Width - rImgStart.X, resizeImage.Height - rImgStart.Y);

            var ratio_x = orgImage.Width / (float)(rImgEnd.X - rImgStart.X);
            var ratio_y = orgImage.Height / (float)(rImgEnd.Y - rImgStart.Y);
            rescaleBox.Xmin = ratio_x * (dBox.Xmin - rImgStart.X);
            rescaleBox.Xmax = ratio_x * (dBox.Xmax - rImgStart.X);
            rescaleBox.Ymin = ratio_y * (dBox.Ymin - rImgStart.Y);
            rescaleBox.Ymax = ratio_y * (dBox.Ymax - rImgStart.Y);
            return rescaleBox;
        }

        private static float[] MatToList(Mat mat)
        {
            var ih = mat.Height;
            var iw = mat.Width;
            var chn = mat.Channels();
            unsafe
            {
                return Create((float*)mat.DataPointer, ih, iw, chn);
            }
        }

        public static Mat CreateLetterbox(Mat img, Size sz, Scalar color, out float ratio, out Point diff, out Point diff2,
            bool auto = true, bool scaleFill = false, bool scaleup = true)
        {
            Mat newImage = new Mat();
            Cv2.CvtColor(img, newImage, ColorConversionCodes.BGR2RGB);
            ratio = Math.Min((float)sz.Width / newImage.Width, (float)sz.Height / newImage.Height);
            if (!scaleup)
            {
                ratio = Math.Min(ratio, 1.0f);
            }
            var newUnpad = new OpenCvSharp.Size((int)Math.Round(newImage.Width * ratio),
                (int)Math.Round(newImage.Height * ratio));
            var dW = sz.Width - newUnpad.Width;
            var dH = sz.Height - newUnpad.Height;

            var tensor_ratio = sz.Height / (float)sz.Width;
            var input_ratio = img.Height / (float)img.Width;
            if (auto && tensor_ratio != input_ratio)
            {
                dW %= 32;
                dH %= 32;
            }
            else if (scaleFill)
            {
                dW = 0;
                dH = 0;
                newUnpad = sz;
            }
            var dW_h = (int)Math.Round((float)dW / 2);
            var dH_h = (int)Math.Round((float)dH / 2);
            var dw2 = 0;
            var dh2 = 0;
            if (dW_h * 2 != dW)
            {
                dw2 = dW - dW_h * 2;
            }
            if (dH_h * 2 != dH)
            {
                dh2 = dH - dH_h * 2;
            }

            if (newImage.Width != newUnpad.Width || newImage.Height != newUnpad.Height)
            {
                Cv2.Resize(newImage, newImage, newUnpad);
            }
            Cv2.CopyMakeBorder(newImage, newImage, dH_h + dh2, dH_h, dW_h + dw2, dW_h, BorderTypes.Constant, color);
            diff = new OpenCvSharp.Point(dW_h, dH_h);
            diff2 = new OpenCvSharp.Point(dw2, dh2);
            return newImage;
        }

        public static List<List<float>> GetCandidate(float[] pred, int[] pred_dim, float pred_thresh = 0.25f)
        {
            List<List<float>> candidate = new List<List<float>>();
            for (int batch = 0; batch < pred_dim[0]; batch++)
            {
                for (int cand = 0; cand < pred_dim[1]; cand++)
                {
                    int score = 4;  // object ness score
                    int idx1 = (batch * pred_dim[1] * pred_dim[2]) + cand * pred_dim[2];
                    int idx2 = idx1 + score;
                    var value = pred[idx2];
                    if (value > pred_thresh)
                    {
                        List<float> tmp_value = new List<float>();
                        for (int i = 0; i < pred_dim[2]; i++)
                        {
                            int sub_idx = idx1 + i;
                            tmp_value.Add(pred[sub_idx]);
                        }
                        candidate.Add(tmp_value);
                    }
                }
            }
            return candidate;
        }

        public static List<List<Mat>> ConvertSegmentationResult(float[] pred, int[] pred_dim, float threshold = 0.25f)
        {
            List<List<Mat>> dataList = new List<List<Mat>>();
            for (int batch = 0; batch < pred_dim[0]; batch++)
            {
                List<Mat> masks = new List<Mat>();
                for(int cls = 0; cls < pred_dim[1]; cls++)
                {
                    List<byte> subData = new List<byte>();
                    for(int h = 0; h < pred_dim[2]; h++)
                    {
                        for(int w = 0; w < pred_dim[3]; w++)
                        {
                            int idx = (batch * pred_dim[1] * pred_dim[2] * pred_dim[3]) +
                                (cls * pred_dim[2] * pred_dim[3]) + (h * pred_dim[3]) + w;
                            if (pred[idx] < threshold)
                            {
                                subData.Add(0);
                            }
                            else
                            {
                                subData.Add(255);
                            }
                        }
                    }
                    using (var mask = new Mat(new int[] { pred_dim[2], pred_dim[3] }, MatType.CV_8UC1, subData.ToArray()))
                    {
                        masks.Add(mask.Clone());
                    }
                }
                dataList.Add(masks);
            }
            return dataList;
        }

        public static List<List<float>> GetSegmentationMatrix(List<List<float>> candidate, int nc,
        float pred_thresh = 0.25f, int max_nms = 30000)
        {
            var mat = new List<List<float>>();
            for (int i = 0; i < candidate.Count; i++)
            {
                if (candidate[i][4] < pred_thresh)
                {
                    // confidence less than threshold
                    continue;
                }

                int cls = -1;
                float max_score = float.MinValue;

                int nc_index_max = 5 + nc;
                for (int j = 5; j < nc_index_max; j++)
                {
                    // find the max score
                    if (candidate[i][j] > max_score)
                    {
                        cls = j;
                        max_score = candidate[i][j];
                    }
                }

                if (cls < 0) continue;

                List<float> tmpDetect = new List<float>();
                for (int j = 0; j < 4; j++) tmpDetect.Add(candidate[i][j]); //box
                tmpDetect.Add(candidate[i][cls]);   //class prob
                tmpDetect.Add(cls - 5);             //class
                mat.Add(tmpDetect);
            }

            //max_nms sort
            mat.Sort((a, b) => (a[4] > b[4]) ? -1 : 1);

            if (mat.Count > max_nms)
            {
                mat.RemoveRange(max_nms, mat.Count - max_nms);
            }
            return mat;
        }

        public static List<List<float>> GetDetectionMatrix(List<List<float>> candidate,
            float pred_thresh = 0.25f, int max_nms = 30000)
        {
            var mat = new List<List<float>>();
            for (int i = 0; i < candidate.Count; i++)
            {
                int cls = -1;
                float max_score = 0;
                for (int j = 5; j < candidate[i].Count; j++)
                {
                    if (candidate[i][j] > pred_thresh && candidate[i][j] >= max_score)
                    {
                        cls = j;
                        max_score = candidate[i][j];
                    }
                }

                if (cls < 0) continue;

                List<float> tmpDetect = new List<float>();
                for (int j = 0; j < 4; j++) tmpDetect.Add(candidate[i][j]); //box
                tmpDetect.Add(candidate[i][cls]);   //class prob
                tmpDetect.Add(cls - 5);             //class
                mat.Add(tmpDetect);
            }

            //max_nms sort
            mat.Sort((a, b) => (a[4] > b[4]) ? -1 : 1);

            if (mat.Count > max_nms)
            {
                mat.RemoveRange(max_nms, mat.Count - max_nms);
            }
            return mat;
        }
    }
}
