use anyhow::Result;
use opencv::{core, highgui, imgcodecs, imgproc, prelude::*, videoio};
use std::f64::consts::PI;

const ASCII_CHARS: &[u8] = b"$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ";
const EDGE_CHARS: &[u8] = b"_/\\|";
const BLOCK_SIZE: u32 = 4;
const GRADIENT_THRESHOLD: f64 = 15.0;

fn main() -> Result<()> {
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    let mut frame = Mat::default();

    highgui::named_window("Combined", highgui::WINDOW_NORMAL)?;

    loop {
        cam.read(&mut frame)?;

        let gaussian = apply_gaussian(&frame)?;
        let gray_image = gray_scale(&gaussian)?;
        let (sobel_image, sobel_data) = apply_sobel(&gray_image)?;
        let down_scaled_image = resize_image(&frame, BLOCK_SIZE)?;

        let ascii_art = convert_to_ascii_art(&down_scaled_image, &sobel_data)?;
        clearscreen::clear().expect("failed to clear screen");
        println!("{}", ascii_art); // Print the ASCII art


        let combined_image = combine_images(&frame, &gaussian, &gray_image, &sobel_image)?;
        highgui::imshow("Combined", &combined_image)?;
        highgui::imshow("Ascii size", &down_scaled_image)?;

        let key = highgui::wait_key(1)?;
        if key == 113 { // quit with q
            break;
        }
    }
    Ok(())
}

// fn main() -> Result<()> {
//     let image_path = "solid-circle.png"; // Change this to your image path
//     let frame = imgcodecs::imread(image_path, imgcodecs::IMREAD_COLOR)?;

//     highgui::named_window("Combined", highgui::WINDOW_NORMAL)?;

//     let gaussian = apply_gaussian(&frame)?;
//     let gray_image = gray_scale(&gaussian)?;
//     let (sobel_image, sobel_data) = apply_sobel(&gray_image)?;
//     let down_scaled_image = resize_image(&frame, BLOCK_SIZE)?;

//     let ascii_art = convert_to_ascii_art(&down_scaled_image, &sobel_data)?;
//     clearscreen::clear().expect("failed to clear screen");
//     println!("{}", ascii_art); // Print the ASCII art

//     let combined_image = combine_images(&frame, &gaussian, &gray_image, &sobel_image)?;
//     loop {
//         highgui::imshow("Combined", &combined_image)?;
//         // highgui::imshow("Combined", &down_scaled_image)?;

//         let key = highgui::wait_key(1)?;
//         if key == 113 { // quit with q
//             break;
//         }
//     }
//     Ok(())
// }

fn  resize_image(image: &Mat, block_size: u32) -> Result<Mat> {
    let size = image.size()?;
    let new_width = size.width / block_size as i32;
    let new_height = size.height / block_size as i32;

    let mut resized_image = Mat::default();
    imgproc::resize(&image, &mut resized_image, core::Size::new(new_width, new_height), 0.0, 0.0, imgproc::INTER_NEAREST)?;
    Ok(resized_image)
}

fn gray_scale(image: &Mat) -> Result<Mat> {
    let mut gray_image = Mat::default();
    imgproc::cvt_color(&image, &mut gray_image, imgproc::COLOR_RGB2GRAY, 0)?;
    Ok(gray_image)
}

fn apply_gaussian(image: &Mat) -> Result<Mat> {
    let mut gaussian = Mat::default();
    imgproc::gaussian_blur(&image, &mut gaussian, core::Size_ { width: 3, height: 3 }, 0.0, 0.0, core::BORDER_DEFAULT)?;
    Ok(gaussian)
}

fn apply_sobel(image: &Mat) -> Result<(Mat, String)> {
    let mut grad_x = Mat::default();
    let mut grad_y = Mat::default();
    imgproc::sobel(&image, &mut grad_x, core::CV_64F, 1, 0, 1, 1.0, 0.0, core::BORDER_DEFAULT)?;
    imgproc::sobel(&image, &mut grad_y, core::CV_64F, 0, 1, 1, 1.0, 0.0, core::BORDER_DEFAULT)?;

    let mut abs_grad_x = Mat::default();
    let mut abs_grad_y = Mat::default();
    core::convert_scale_abs(&grad_x, &mut abs_grad_x, 1.0, 0.0)?;
    core::convert_scale_abs(&grad_y, &mut abs_grad_y, 1.0, 0.0)?;

    let mut grad = Mat::default();
    core::add_weighted(&abs_grad_x, 0.8, &abs_grad_y, 0.8, 0.0, &mut grad, -1)?;

    let (width, height) = (grad.cols(), grad.rows());

    let threshold = BLOCK_SIZE / 2 - 1;

    let mut ascii_art = String::new();
    for y in (0..height).step_by(BLOCK_SIZE as usize) {
        for x in (0..width).step_by(BLOCK_SIZE as usize) {
            // Init list of counts
            let mut counter = [0; 4];

            // Go through all the different pixels in a grid of BLOCK_SIZE * BLOCK_SIZE
            for ky in 0..BLOCK_SIZE as i32 {
                for kx in 0..BLOCK_SIZE as i32{
                    // check if we are within the image
                    if x + kx < width && y + ky < height {
                        let gx = *grad_x.at_2d::<f64>(y + ky as i32, x + kx as i32)?;
                        let gy = *grad_y.at_2d::<f64>(y + ky as i32, x + kx as i32)?;

                        // Calculate the edgechar count
                        let magnitude = gx.hypot(gy);
                        if magnitude > GRADIENT_THRESHOLD {
                            let angle = (gy.atan2(gx) * (180.0 / PI)).abs(); // between 0-180 degrees

                            let ascii_char_index: usize;
                            // Get the ascii char index that correspond to the pixel angle
                            if gy > 0.0 {
                                ascii_char_index = match angle {
                                    a if a <= 15.0 || (165.0 < a && a <= 180.0) => 3,   // _
                                    a if 15.0 < a && a <= 75.0 => 1, // /
                                    a if 75.0 < a && a <= 105.0 => 0, // |
                                    _ => 2,             // \
                                };
                            } else {
                                ascii_char_index = match angle {
                                    a if a <= 22.5 || (157.5 < a && a <= 180.0) => 3,   // _
                                    a if 22.5 < a && a <= 67.5 => 2, // /
                                    a if 67.5 < a && a <= 112.5 => 0, // |
                                    _ => 1,             // \
                                };
                            }

                            // Add the ascii char to the counter
                            counter[ascii_char_index] += 1;
                        }
                    }
                }
            }

            // Find the index with the maximum value in the counter
            let (ascii_char_index, &max_value) = counter.iter().enumerate().max_by_key(|&(_, &count)| count).unwrap();

            // Look up if the max value is bigger than the threshold
            if max_value > threshold {
                // Get the ascii character based on the max values index
                let ascii_char = EDGE_CHARS[ascii_char_index];
                ascii_art.push(ascii_char as char);
            } else {
                ascii_art.push(' '); // Use space for insignificant gradients
            }
        }
        ascii_art.push('\n');
    }

    Ok((grad, ascii_art))
}

fn convert_to_ascii_art(frame: &Mat, sobel_data: &str) -> Result<String> {
    let (width, height) = (frame.cols(), frame.rows());

    // Check if the size is equal
    if sobel_data.len() != (width * height + height) as usize {
        panic!("The data length is not correct!")
    }

    let sobel_chars: Vec<char> = sobel_data.chars().collect();

    let ascii_art: String = (0..height).into_iter().map(|y| {
        let mut line = String::new();
        for x in 0..width {
            // Get the color data
            let pixel = frame.at_2d::<core::Vec3b>(y, x).unwrap();
            let red = pixel[2]; // OpenCV uses BGR format
            let green = pixel[1];
            let blue = pixel[0];

            let index = (y * width + x + y) as usize;
            if sobel_chars[index] == ' ' {
                let luminance = 0.2126 * red as f32 + 0.7152 * green as f32 + 0.0722 * blue as f32;
                let char_index = (luminance / 255.0 * (ASCII_CHARS.len() - 1) as f32).round() as usize;
                let ascii_char = ASCII_CHARS[char_index];
                line.push_str(&format!("\x1b[38;2;{};{};{}m{} ", red, green, blue, ascii_char as char));
            } else {
                line.push_str(&format!("\x1b[38;2;{};{};{}m{} ", red, green, blue, sobel_chars[index]));
            }
        }
        line.push('\n');
        line
    }).collect();

    Ok(ascii_art)
}

fn combine_images(
    original: &Mat,
    gaussian: &Mat,
    gray: &Mat,
    sobel: &Mat,
) -> Result<Mat> {
    let original_size = original.size()?;
    let width = original_size.width * 2;
    let height = original_size.height * 2;

    let mut combined = Mat::new_rows_cols_with_default(height, width, original.typ(), core::Scalar::default())?;

    // Copy original image to top-left quadrant
    let rect = core::Rect::new(0, 0, original_size.width, original_size.height);
    let mut roi = Mat::roi_mut(&mut combined, rect)?;
    original.copy_to(&mut roi)?;

    // Copy Gaussian image to top-right quadrant (no conversion needed)
    let rect = core::Rect::new(original_size.width, 0, original_size.width, original_size.height);
    let mut roi = Mat::roi_mut(&mut combined, rect)?;
    gaussian.copy_to(&mut roi)?;

    // Copy grayscale image to bottom-left quadrant (convert to BGR for display)
    let rect = core::Rect::new(0, original_size.height, original_size.width, original_size.height);
    let mut roi = Mat::roi_mut(&mut combined, rect)?;
    imgproc::cvt_color(&gray, &mut roi, imgproc::COLOR_GRAY2BGR, 0)?;

    // Copy Sobel image to bottom-right quadrant (convert to BGR for display)
    let rect = core::Rect::new(original_size.width, original_size.height, original_size.width, original_size.height);
    let mut roi = Mat::roi_mut(&mut combined, rect)?;
    imgproc::cvt_color(&sobel, &mut roi, imgproc::COLOR_GRAY2BGR, 0)?;

    Ok(combined)
}
