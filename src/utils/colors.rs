use derivative::Derivative;
use image::imageops::overlay;
use image::{DynamicImage, GenericImageView, Rgb, Rgba, RgbaImage};
use imageproc::filter::gaussian_blur_f32;
use ndarray;
use onnxruntime::session::Session;
use onnxruntime::{
    environment::Environment, ndarray::Array4, tensor::OrtOwnedTensor, GraphOptimizationLevel,
};
use serenity::all::{ButtonStyle, CreateActionRow, CreateButton, ReactionType};
use std::collections::HashMap;
use std::fmt::Display;
use std::num::ParseIntError;
use std::vec;

use crate::config::load_config;

#[derive(Clone, Debug, Default)]
pub enum ImageType {
    Cartoon,
    #[default]
    Picture,
}

#[derive(Clone, Debug, Copy, PartialEq, Default)]
pub enum Models {
    U2net,
    IsnetAnime,
    IsnetGeneral,
    #[default]
    Algorithm,
}

#[derive(Clone, Debug)]
pub struct Model {
    pub id: usize,
    pub path: String,
    pub name: String,
    pub width: u32,
    pub height: u32,
}

impl Models {
    pub fn to_struct(&self) -> Model {
        let pwd: String = load_config().threshold.modelpath.into();
        match self {
            Models::U2net => Model {
                id: 0,
                path: String::from(format!("{pwd}/u2net.onnx")),
                name: String::from("AI General 2"),
                width: 320,
                height: 320,
            },
            Models::IsnetAnime => Model {
                id: 1,
                path: String::from(format!("{pwd}/isnet-anime.onnx")),
                name: String::from("AI Anime"),
                width: 1024,
                height: 1024,
            },
            Models::IsnetGeneral => Model {
                id: 2,
                path: String::from(format!("{pwd}/isnet-general-use.onnx")),
                name: String::from("AI General"),
                width: 1024,
                height: 1024,
            },
            Models::Algorithm => Model {
                id: 3,
                path: String::from("LOCAL"),
                name: String::from("General"),
                width: 320,
                height: 320,
            },
        }
    }

    pub fn from_id(id: usize) -> Self {
        match id {
            0 => Models::U2net,
            1 => Models::IsnetAnime,
            2 => Models::IsnetGeneral,
            3 => Models::Algorithm,
            _ => Models::Algorithm,
        }
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Default)]
pub enum ActivationFunction {
    #[default]
    Linear,
    Sigmoid,
    ReLU,
    Tanh,
    Softmax,
}
impl ActivationFunction {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(ActivationFunction::Linear),
            1 => Some(ActivationFunction::Sigmoid),
            2 => Some(ActivationFunction::ReLU),
            3 => Some(ActivationFunction::Tanh),
            4 => Some(ActivationFunction::Softmax),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            ActivationFunction::Linear => "Linear",
            ActivationFunction::Sigmoid => "Sigmoid",
            ActivationFunction::ReLU => "ReLU",
            ActivationFunction::Tanh => "Tanh",
            ActivationFunction::Softmax => "Softmax",
        }
    }

    pub fn next(&self) -> Self {
        let values = vec![
            ActivationFunction::Linear,
            ActivationFunction::Sigmoid,
            // ActivationFunction::ReLU, ActivationFunction::Tanh,
            // ActivationFunction::Softmax
        ];
        let self_index = values.iter().position(|&x| x == *self).unwrap();
        let next = (self_index + 1) % (values.len());
        values[next]
    }
}
// implement clone
#[derive(Clone, Debug)]
pub enum NordPreset {
    NordWithColor,
    Nord,
    StaticBackground,
    DynamicBackground,
}

impl NordPreset {
    pub fn iter() -> Vec<NordPreset> {
        vec![
            NordPreset::NordWithColor,
            NordPreset::Nord,
            NordPreset::StaticBackground,
            NordPreset::DynamicBackground,
        ]
    }
}

#[derive(Clone, Debug, Derivative, Default)]
#[derivative(PartialEq)]
pub struct NordOptions {
    pub invert: bool,
    pub hue_rotate: f32,
    pub sepia: bool,
    pub nord: bool,
    pub erase_most_present_color: bool,

    #[derivative(PartialEq = "ignore")]
    pub erase_when_percentage: f64,

    #[derivative(PartialEq = "ignore")]
    pub auto_adjust: bool,

    #[derivative(PartialEq = "ignore")]
    pub start: bool,

    pub model: Models,
    pub activation_function: ActivationFunction,
    pub background_color: Option<RgbColor>,

    #[derivative(PartialEq = "ignore")]
    pub simple_layout: bool,
}

impl NordOptions {
    pub fn new() -> Self {
        let mut options = NordOptions::default();
        options.start = true;
        options
    }
    pub fn default() -> Self {
        NordOptions {
            invert: true,
            hue_rotate: 180.0,
            sepia: true,
            nord: true,
            erase_most_present_color: false,
            erase_when_percentage: 0.3, // if met: all other filters are ignored
            auto_adjust: true,
            start: false,
            model: Models::Algorithm,
            activation_function: ActivationFunction::Sigmoid,
            background_color: None,
            simple_layout: true,
        }
    }

    pub fn from_image_information(image_information: &ImageInformation) -> Self {
        let mut options = NordOptions::default();
        let invert_by_brightness = image_information.brightness.average > 0.5;
        let is_probably_anime = |info: &ImageInformation| -> bool {
            info.color_map.most_present_color_percentage > 0.005
                && info.grayscale_similarity.average > 0.06
        };
        match image_information.image_type {
            Some(ImageType::Cartoon) => {
                options.erase_most_present_color = false; //image_information.color_map.most_present_color_percentage > 0.1;
                options.invert = invert_by_brightness;
                options.hue_rotate = 180.;
                options.sepia = true;
                options.nord = true;
                options.erase_when_percentage = 0.1;
                options.auto_adjust = false;
                options.start = false;
                options.model = Models::Algorithm;
                options.background_color = None;
            }
            Some(ImageType::Picture) => {
                if image_information.color_map.most_present_color_percentage > 0.1 {
                    // image, but replaced monotone background
                    options.invert = false;
                    options.hue_rotate = 0.;
                    options.sepia = false;
                    options.nord = false;
                    options.erase_most_present_color = true;
                    options.erase_when_percentage = 0.1;
                    options.auto_adjust = false;
                    options.start = false;
                    options.model = if is_probably_anime(image_information) {
                        Models::IsnetAnime
                    } else {
                        Models::IsnetGeneral
                    };
                    options.background_color = None;
                } else {
                    // image without predominant color
                    options.invert = false;
                    options.hue_rotate = 0.;
                    options.sepia = false;
                    options.nord = false;
                    options.erase_most_present_color = true;
                    options.erase_when_percentage = 0.1;
                    options.auto_adjust = false;
                    options.start = false;
                    options.model = if is_probably_anime(image_information) {
                        Models::IsnetAnime
                    } else {
                        Models::IsnetGeneral
                    };
                    options.background_color = None;
                }
            }
            None => {}
        }
        options
    }

    pub fn from_preset(preset: NordPreset, nord_options: &NordOptions) -> NordOptions {
        match preset {
            NordPreset::NordWithColor => NordOptions {
                sepia: false,
                auto_adjust: false,
                simple_layout: nord_options.simple_layout,
                ..NordOptions::default()
            },
            NordPreset::Nord => NordOptions {
                auto_adjust: false,
                simple_layout: nord_options.simple_layout,
                ..NordOptions::default()
            },
            NordPreset::StaticBackground => NordOptions {
                invert: false,
                hue_rotate: 0.0,
                sepia: false,
                nord: false,
                erase_most_present_color: true,
                erase_when_percentage: 0.1,
                auto_adjust: false,
                start: false,
                model: Models::Algorithm,
                activation_function: ActivationFunction::Sigmoid,
                background_color: None,
                ..nord_options.clone()
            },
            NordPreset::DynamicBackground => NordOptions {
                invert: false,
                hue_rotate: 0.0,
                sepia: false,
                nord: false,
                erase_most_present_color: true,
                erase_when_percentage: 0.1,
                auto_adjust: false,
                start: false,
                model: Models::IsnetGeneral,
                activation_function: ActivationFunction::Sigmoid,
                background_color: None,
                ..nord_options.clone()
            },
        }
    }

    pub fn is_any_preset(&self) -> bool {
        for preset in NordPreset::iter() {
            if self == &NordOptions::from_preset(preset, &NordOptions::default()) {
                return true;
            }
        }
        false
    }

    pub fn is_preset(&self, preset: NordPreset) -> bool {
        self == &NordOptions::from_preset(preset, &NordOptions::default())
    }

    pub fn make_nord_custom_id(&self, message_id: &u64, update: bool, id: Option<usize>) -> String {
        // id is needed to make the custom id unique since there could be buttons which do the same
        format!(
            "darken-{}-{}-{}-{}-{}-{}-{:.2}-{}-{}-{}-{}-{}-{}-{}-{}",
            update,
            self.invert,
            self.hue_rotate,
            self.sepia,
            self.nord,
            self.erase_most_present_color,
            self.erase_when_percentage,
            self.auto_adjust,
            self.start,
            self.model.to_struct().id,
            self.activation_function as u8,
            id.unwrap_or(0),
            if self.background_color.is_some() {
                self.background_color.unwrap().as_hex()
            } else {
                "None".to_string()
            },
            self.simple_layout,
            message_id,
        )
    }

    pub fn from_custom_id(custom_id: &str) -> Self {
        let mut parts = custom_id.split("-").skip(1);
        let _update = parts.next().unwrap().parse::<bool>().unwrap();
        let invert = parts.next().unwrap().parse::<bool>().unwrap();
        let hue_rotate = parts.next().unwrap().parse::<f32>().unwrap();
        let sepia = parts.next().unwrap().parse::<bool>().unwrap();
        let nord = parts.next().unwrap().parse::<bool>().unwrap();
        let erase_most_present_color = parts.next().unwrap().parse::<bool>().unwrap();
        let erase_when_percentage = parts.next().unwrap().parse::<f64>().unwrap();
        let auto_adjust = parts.next().unwrap().parse::<bool>().unwrap();
        let start = parts.next().unwrap().parse::<bool>().unwrap();
        let model_id: usize = parts.next().unwrap().parse::<usize>().unwrap();
        let model = Models::from_id(model_id);
        let activation_function_id = parts.next().unwrap().parse::<u8>().unwrap();
        let activation_function = ActivationFunction::from_u8(activation_function_id).expect(
            &format!("Invalid ActivationFunction ID: {}", activation_function_id),
        );
        let _id = parts.next().unwrap().parse::<usize>().unwrap();
        // colors in format r;g;b
        let background_color_str: &str = parts.next().unwrap();
        let background_color = if background_color_str == "None" {
            None
        } else {
            Some(RgbColor::from_hex(background_color_str).unwrap())
        };
        let simple_layout = parts.next().unwrap().parse::<bool>().unwrap();
        let _message_id = parts.next().unwrap().parse::<u64>().unwrap();
        NordOptions {
            invert,
            hue_rotate,
            sepia,
            nord,
            erase_most_present_color,
            erase_when_percentage,
            auto_adjust,
            start,
            model,
            activation_function,
            background_color,
            simple_layout,
        }
    }

    pub fn modal_get_color(&self) {}
    pub fn build_componets(&self, message_id: u64, update: bool) -> Vec<CreateActionRow> {
        let mut components = Vec::new();
        let mut action_rows = Vec::<Vec<CreateButton>>::new();
        let mut self_no_start = self.clone();
        self_no_start.start = false;

        println!("make components with bg: {:?}", self.background_color);
        // make option lists, so that the clicked button is inverted
        let option_2d_list = match self.simple_layout {
            true => self._generate_simple_compoenents(),
            false => self._generate_detailed_components(),
        };

        let mut name_to_color_map = HashMap::<&str, ButtonStyle>::new();
        name_to_color_map.insert("Start", ButtonStyle::Success);

        // convert vec to components
        for (x, option_list) in option_2d_list.into_iter().enumerate() {
            if option_list.len() == 0 {
                continue;
            }
            let mut action_row = Vec::<CreateButton>::new();
            for (y, (label, enabled, option, is_enabled)) in option_list.into_iter().enumerate() {
                // iterate over one inner vec
                action_row.push(
                    CreateButton::new(option.make_nord_custom_id(
                        &message_id,
                        update,
                        Some(x * 10 + y),
                    ))
                    .label(&format!("{}", label))
                    .style({
                        *name_to_color_map.get(label.as_str()).unwrap_or(if enabled {
                            &ButtonStyle::Primary
                        } else {
                            &ButtonStyle::Secondary
                        })
                    })
                    .disabled(!is_enabled),
                );
            }
            action_rows.push(action_row);
        }
        for action_row in action_rows {
            components.push(CreateActionRow::Buttons(action_row));
        }
        let mut last_row: Vec<CreateButton> = vec![
            CreateButton::new(format!("delete-{}", message_id))
                .style(ButtonStyle::Secondary)
                .label("Keep new")
                .emoji("✅".parse::<ReactionType>().unwrap()),
            // stop button
            CreateButton::new(format!("stop-{}", message_id))
                .style(ButtonStyle::Secondary)
                .label("Keep old")
                .emoji("✅".parse::<ReactionType>().unwrap()),
            CreateButton::new(format!("clear-{}", message_id))
                .style(ButtonStyle::Secondary)
                .emoji("✅".parse::<ReactionType>().unwrap())
                .label("Keep both"),
        ];
        // add start button
        if !self.start {
            last_row.insert(
                0,
                CreateButton::new(
                    NordOptions {
                        start: !self.start,
                        ..self_no_start
                    }
                    .make_nord_custom_id(&message_id, update, Some(51)),
                )
                .style(ButtonStyle::Success)
                .label("Start")
                .emoji("▶️".parse::<ReactionType>().unwrap()),
            );
        }
        components.push(CreateActionRow::Buttons(last_row));

        components
    }

    fn _generate_simple_compoenents(&self) -> Vec<Vec<(String, bool, NordOptions, bool)>> {
        let mut self_no_start = self.clone();
        self_no_start.start = false;
        let layout_label = if !self.simple_layout {
            "Simple Layout"
        } else {
            "Advanced Layout"
        };

        // make option lists, so that the clicked button is inverted
        let option_2d_list: Vec<Vec<(String, bool, NordOptions, bool)>> = vec![
            vec![(
                "▼ More Options".into(),
                !self.simple_layout,
                NordOptions {
                    simple_layout: !self.simple_layout,
                    ..self_no_start
                },
                true,
            )],
            // preset vec
            vec![
                (
                    "Colorful Dark".into(),
                    self.is_preset(NordPreset::NordWithColor),
                    NordOptions::from_preset(NordPreset::NordWithColor, &self_no_start),
                    true,
                ),
                (
                    "Mono Dark".into(),
                    self.is_preset(NordPreset::Nord),
                    NordOptions::from_preset(NordPreset::Nord, &self_no_start),
                    true,
                ),
                (
                    "Static Background".into(),
                    self.is_preset(NordPreset::StaticBackground),
                    NordOptions::from_preset(NordPreset::StaticBackground, &self_no_start),
                    true,
                ),
                (
                    "Dynamic Background".into(),
                    self.is_preset(NordPreset::DynamicBackground),
                    NordOptions::from_preset(NordPreset::DynamicBackground, &self_no_start),
                    true,
                ),
            ],
        ];
        option_2d_list
    }

    fn _generate_detailed_components(&self) -> Vec<Vec<(String, bool, NordOptions, bool)>> {
        let mut self_no_start = self.clone();
        self_no_start.start = false;

        let is_model_enabled = |x: &Self| x.erase_most_present_color;
        let background_color = if self.background_color.is_some() {
            self.background_color.unwrap().to_string()
        } else {
            "None".to_owned()
        };
        let function_name = format!("Mask Function: {}", self.activation_function.as_str());
        println!("make components with bg: {:?}", self.background_color);
        // make option lists, so that the clicked button is inverted
        let option_2d_list: Vec<Vec<(String, bool, NordOptions, bool)>> = vec![
            // component row
            vec![
                // component
                //name: intert, blue/gray, When click, then switch enabled/disabled, is enabled // arrow up str: ▲ // arrow down str: ▼
                (
                    "▲ Show only Presets".into(),
                    !self.simple_layout,
                    NordOptions {
                        simple_layout: !self.simple_layout,
                        ..self_no_start
                    },
                    true,
                ),
                (
                    "Invert".into(),
                    self.invert,
                    NordOptions {
                        invert: !self.invert,
                        ..self_no_start
                    },
                    true,
                ),
                (
                    "Hue Rotate".into(),
                    if self.hue_rotate == 180. { true } else { false },
                    NordOptions {
                        hue_rotate: if self.hue_rotate == 180. { 0. } else { 180. },
                        ..self_no_start
                    },
                    true,
                ),
                (
                    "Sepia".into(),
                    self.sepia,
                    NordOptions {
                        sepia: !self.sepia,
                        ..self_no_start
                    },
                    true,
                ),
                (
                    "Nord".into(),
                    self.nord,
                    NordOptions {
                        nord: !self.nord,
                        ..self_no_start
                    },
                    true,
                ),
            ],
            vec![
                (
                    "Erase Background".into(),
                    self.erase_most_present_color,
                    NordOptions {
                        erase_most_present_color: !self.erase_most_present_color,
                        ..self_no_start
                    },
                    true,
                ),
                (
                    "Dominant Color".into(),
                    self.model == Models::Algorithm,
                    NordOptions {
                        model: Models::Algorithm,
                        ..self_no_start
                    },
                    is_model_enabled(self),
                ),
                (
                    "General Use".into(),
                    self.model == Models::IsnetGeneral,
                    NordOptions {
                        model: Models::IsnetGeneral,
                        ..self_no_start
                    },
                    is_model_enabled(self),
                ),
                //("General Use 2", self.model == Models::U2net, NordOptions {model: Models::U2net, ..self_no_start}, is_model_enabled(self)),
                (
                    "Anime".into(),
                    self.model == Models::IsnetAnime,
                    NordOptions {
                        model: Models::IsnetAnime,
                        ..self_no_start
                    },
                    is_model_enabled(self),
                ),
                (
                    function_name,
                    true,
                    NordOptions {
                        activation_function: self.activation_function.next(),
                        ..self_no_start
                    },
                    is_model_enabled(self),
                ),
            ],
            vec![
                (
                    "Set Background".into(),
                    self.background_color.is_some(),
                    NordOptions {
                        background_color: if self.background_color.is_some() {
                            None
                        } else {
                            Some(RgbColor::from_hex("424242").unwrap())
                        },
                        ..self_no_start
                    },
                    true,
                ),
                (
                    background_color,
                    self.background_color.is_some(),
                    NordOptions {
                        background_color: Some(RgbColor::from_hex("000001").unwrap()),
                        ..self_no_start
                    },
                    self.background_color.is_some(),
                ), // 000001 is reserved for setting new color
            ],
            // preset vec
            vec![
                //("Presets:".into(), self.is_any_preset(), NordOptions { ..self_no_start}, false),
                (
                    "Nord w/ Color".into(),
                    self.is_preset(NordPreset::NordWithColor),
                    NordOptions::from_preset(NordPreset::NordWithColor, &self_no_start),
                    true,
                ),
                (
                    "Nord w/o Color".into(),
                    self.is_preset(NordPreset::Nord),
                    NordOptions::from_preset(NordPreset::Nord, &self_no_start),
                    true,
                ),
                (
                    "Static Background".into(),
                    self.is_preset(NordPreset::StaticBackground),
                    NordOptions::from_preset(NordPreset::StaticBackground, &self_no_start),
                    true,
                ),
                (
                    "Dynamic Background".into(),
                    self.is_preset(NordPreset::DynamicBackground),
                    NordOptions::from_preset(NordPreset::DynamicBackground, &self_no_start),
                    true,
                ),
            ],
        ];
        option_2d_list
    }
}

#[derive(Clone, Debug, PartialEq, Copy)]
pub struct RgbColor {
    r: u8,
    g: u8,
    b: u8,
}

impl RgbColor {
    pub fn rn(&self) -> f32 {
        self.r as f32 / 255.0
    }

    pub fn gn(&self) -> f32 {
        self.g as f32 / 255.0
    }

    pub fn bn(&self) -> f32 {
        self.b as f32 / 255.0
    }

    pub fn brightness(&self) -> f32 {
        (0.299 * self.r as f32 + 0.587 * self.g as f32 + 0.114 * self.b as f32) / 255.0
    }

    pub fn calculate_grayscale_similarity(&self) -> f32 {
        let r_f32 = self.rn();
        let g_f32 = self.gn();
        let b_f32 = self.bn();

        // Calculate the mean of the RGB values
        let mean = (r_f32 + g_f32 + b_f32) / 3.0;

        // Calculate the squared differences from the mean
        let r_diff = (r_f32 - mean).powi(2);
        let g_diff = (g_f32 - mean).powi(2);
        let b_diff = (b_f32 - mean).powi(2);

        // Calculate the variance (mean of squared differences)
        let variance = (r_diff + g_diff + b_diff) / 3.0;

        // The standard deviation is the square root of the variance
        let std_dev = variance.sqrt();

        std_dev
    }
    fn darken_rgb(&self, amount: f32) -> RgbColor {
        // Clamp RGB values between 0 and 1
        // Calculate darkened RGB values
        let new_r = self.rn() - amount;
        let new_g = self.gn() - amount;
        let new_b = self.bn() - amount;

        // Clamp darkened RGB values between 0 and 1
        let new_r = new_r.max(0.0).min(1.0);
        let new_g = new_g.max(0.0).min(1.0);
        let new_b = new_b.max(0.0).min(1.0);

        RgbColor {
            r: (new_r * 255.0) as u8,
            g: (new_g * 255.0) as u8,
            b: (new_b * 255.0) as u8,
        }
    }
    fn color_distance(&self, c2: (u8, u8, u8)) -> f32 {
        let (r1, g1, b1) = (self.r, self.g, self.b);
        let (r2, g2, b2) = c2;
        let dr = r1 as f32 - r2 as f32;
        let dg = g1 as f32 - g2 as f32;
        let db = b1 as f32 - b2 as f32;
        (dr * dr + dg * dg + db * db).sqrt()
    }

    pub fn from_hex(hex: &str) -> Result<Self, ParseIntError> {
        let hex = hex.trim_start_matches("#");
        let r = u8::from_str_radix(&hex[0..2], 16)?;
        let g = u8::from_str_radix(&hex[2..4], 16)?;
        let b = u8::from_str_radix(&hex[4..6], 16)?;
        Ok(RgbColor { r, g, b })
    }

    pub fn as_hex(&self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
    }
}

impl Display for RgbColor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "#{:02x}{:02x}{:02x} (r: {} g: {} b {})",
            self.r, self.g, self.b, self.r, self.g, self.b
        )
    }
}

struct PolarNight {}
impl PolarNight {
    const A: RgbColor = RgbColor {
        r: 46,
        g: 52,
        b: 64,
    };
    const B: RgbColor = RgbColor {
        r: 59,
        g: 66,
        b: 82,
    };
    const C: RgbColor = RgbColor {
        r: 67,
        g: 76,
        b: 94,
    };
    const D: RgbColor = RgbColor {
        r: 76,
        g: 86,
        b: 106,
    };
}

// struct SnowStorm {}
// impl SnowStorm {
//     const A: RgbColor = RgbColor {r: 216, g: 222, b: 233};
//     const B: RgbColor = RgbColor {r: 229, g: 233, b: 240};
//     const C: RgbColor = RgbColor {r: 236, g: 239, b: 244};
//     const D: RgbColor = RgbColor {r: 236, g: 239, b: 244};
// }

struct Frost {}
// impl for #8fbcbb #88c0d0 #81a1c1 #5e81ac
impl Frost {
    const A: RgbColor = RgbColor {
        r: 143,
        g: 188,
        b: 187,
    };
    const B: RgbColor = RgbColor {
        r: 136,
        g: 192,
        b: 208,
    };
    const C: RgbColor = RgbColor {
        r: 129,
        g: 161,
        b: 193,
    };
    const D: RgbColor = RgbColor {
        r: 94,
        g: 129,
        b: 172,
    };
}

pub fn apply_nord(
    mut _image: DynamicImage,
    options: NordOptions,
    info: &ImageInformation,
) -> DynamicImage {
    let mut image = _image.clone();
    println!("{:?}", image.dimensions());
    //image = image.grayscale();
    println!("Brightness of image is: {:.3}", info.brightness.average);

    if options.erase_most_present_color {
        if options.model != Models::Algorithm {
            // Remove background with AI
            // load AI model
            let model_path = options.model.to_struct().path;
            let environment = Environment::builder()
                .with_name("background_removal")
                .with_log_level(onnxruntime::LoggingLevel::Warning)
                .build()
                .unwrap();

            let session = environment
                .new_session_builder()
                .unwrap()
                .with_optimization_level(GraphOptimizationLevel::Basic)
                .unwrap()
                .with_model_from_file(model_path)
                .unwrap();

            let start = std::time::Instant::now();
            let segmented_image = remove_background(session, image, &options);
            println!(
                "[Total] Time taken: {:.3} seconds",
                start.elapsed().as_secs_f32()
            );
            image = segmented_image;
        } else {
            //Remove most present color if above threshold
            let mut mod_image = image.to_rgba8();
            let (most_present_color_tuple, percentage) = (
                info.color_map.most_present_color,
                info.color_map.most_present_color_percentage,
            );
            let most_present_color = RgbColor {
                r: most_present_color_tuple.0,
                g: most_present_color_tuple.1,
                b: most_present_color_tuple.2,
            };
            if percentage >= options.erase_when_percentage {
                // there is actually a color to remove -> remove it
                remove_most_present_colors(&mut mod_image, most_present_color, 40.);
                image = DynamicImage::from(mod_image);
            }
        }
    }

    if options.invert {
        image.invert();
    }
    let mut mod_image = image.to_rgba8();
    if options.sepia {
        apply_sepia(&mut mod_image);
    }
    if options.hue_rotate != 0.0 {
        mod_image = DynamicImage::from(mod_image)
            .huerotate(options.hue_rotate as i32)
            .to_rgba8();
    }
    if options.nord {
        apply_nord_filter(&mut mod_image, &options);
    }
    if options.background_color.is_some() {
        let background_color = options.background_color.unwrap();
        let mut background_image = RgbaImage::from_pixel(
            image.width(),
            image.height(),
            Rgba([
                background_color.r,
                background_color.g,
                background_color.b,
                255,
            ]),
        );
        overlay(&mut background_image, &image, 0, 0);
        mod_image = background_image;
    }
    if options.sepia
        || options.hue_rotate != 0.0
        || options.nord
        || options.background_color.is_some()
    {
        DynamicImage::from(mod_image)
    } else {
        image
    }
}

pub fn _tint_image(image: &mut RgbaImage, tint: Rgb<f32>) {
    let Rgb([tint_r, tint_g, tint_b]) = tint;
    for Rgba([r, g, b, _]) in image.pixels_mut() {
        *r = (*r as f32 * tint_r) as u8;
        *g = (*g as f32 * tint_g) as u8;
        *b = (*b as f32 * tint_b) as u8;
    }
}

pub fn apply_sepia(image: &mut RgbaImage) {
    for Rgba([r, g, b, _]) in image.pixels_mut() {
        let tr = (0.393 * *r as f32 + 0.769 * *g as f32 + 0.189 * *b as f32).min(255.0) as u8;
        let tg = (0.349 * *r as f32 + 0.686 * *g as f32 + 0.168 * *b as f32).min(255.0) as u8;
        let tb = (0.272 * *r as f32 + 0.534 * *g as f32 + 0.131 * *b as f32).min(255.0) as u8;
        *r = tr;
        *g = tg;
        *b = tb;
    }
}

pub fn _apply_tone(image: &mut RgbaImage, target_color: Rgb<f32>, blend_factor: f32) {
    let Rgb([target_r, target_g, target_b]) = target_color;
    for Rgba([r, g, b, _]) in image.pixels_mut() {
        let orig_r = *r as f32 / 255.0;
        let orig_g = *g as f32 / 255.0;
        let orig_b = *b as f32 / 255.0;

        *r = ((orig_r * (1.0 - blend_factor) + target_r * blend_factor) * 255.0).min(255.0) as u8;
        *g = ((orig_g * (1.0 - blend_factor) + target_g * blend_factor) * 255.0).min(255.0) as u8;
        *b = ((orig_b * (1.0 - blend_factor) + target_b * blend_factor) * 255.0).min(255.0) as u8;
    }
}

pub fn calculate_average_brightness(image: &RgbaImage) -> ImageInformation {
    let image_information = get_image_information(&image);
    println!(
        "--------------- IMAGE INFORMATION -------------\n{:?}",
        image_information
    );
    image_information
}

pub fn apply_nord_filter(image: &mut RgbaImage, options: &NordOptions) {
    let mut smallest_grey = f32::MAX;
    let mut biggest_grey = f32::MIN;
    let max_brightness = if options.erase_most_present_color {
        1.
    } else {
        0.85
    };

    let contrast_colors = vec![PolarNight::A, PolarNight::B, PolarNight::C, PolarNight::D];

    let colorful_colors = vec![Frost::A, Frost::B, Frost::C, Frost::D];

    for color in &contrast_colors {
        println!(
            "{} {} {} has brightness {:.3}",
            color.r,
            color.g,
            color.b,
            color.brightness()
        );
    }

    fn get_nearest_color<'a>(color: &RgbColor, all_colors: &'a [RgbColor]) -> &'a RgbColor {
        let mut min_distance = f32::MAX;
        let mut nearest_color = &all_colors[0];
        let br = color.brightness();
        for c in all_colors.iter() {
            let dist = (c.brightness() - br).abs();
            if dist < min_distance {
                min_distance = dist;
                nearest_color = c;
            }
        }
        nearest_color
    }

    let mut cache: HashMap<(u8, u8, u8), (u8, u8, u8)> = HashMap::new();

    for Rgba([r, g, b, _]) in image.pixels_mut() {
        let key = (*r, *g, *b);
        if let Some(&(cached_r, cached_g, cached_b)) = cache.get(&key) {
            *r = cached_r;
            *g = cached_g;
            *b = cached_b;
            continue;
        }

        let color = RgbColor {
            r: *r,
            g: *g,
            b: *b,
        };
        let current_pixel_br = color.brightness();
        let grayscale_similarity = color.calculate_grayscale_similarity();

        if grayscale_similarity < smallest_grey {
            smallest_grey = grayscale_similarity;
        }
        if grayscale_similarity > biggest_grey {
            biggest_grey = grayscale_similarity;
        }

        let darken_by = (current_pixel_br - max_brightness).max(0.0);
        let adjusted_color = if darken_by > 0.0 {
            color.darken_rgb(darken_by)
        } else {
            color
        };

        let nearest_color = if grayscale_similarity < 0.25 {
            get_nearest_color(&adjusted_color, &contrast_colors)
        } else {
            get_nearest_color(&adjusted_color, &colorful_colors)
        };

        let strength = (1.0 - (current_pixel_br - nearest_color.brightness()).abs()) * 0.8;

        let blended_r =
            (adjusted_color.rn() * (1.0 - strength) + nearest_color.rn() * strength) * 255.0;
        let blended_g =
            (adjusted_color.gn() * (1.0 - strength) + nearest_color.gn() * strength) * 255.0;
        let blended_b =
            (adjusted_color.bn() * (1.0 - strength) + nearest_color.bn() * strength) * 255.0;

        let final_r = blended_r.min(255.0) as u8;
        let final_g = blended_g.min(255.0) as u8;
        let final_b = blended_b.min(255.0) as u8;

        cache.insert(key, (final_r, final_g, final_b));

        *r = final_r;
        *g = final_g;
        *b = final_b;
    }

    println!("greyscale: {:.3} - {:.3}", smallest_grey, biggest_grey);
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn map_distance_to_transparency(distance: f32, max_distance: f32) -> u8 {
    if distance >= max_distance {
        0
    } else {
        let alpha = smoothstep(0.0, max_distance, distance);
        (alpha * 255.0) as u8
    }
}

pub fn remove_most_present_colors(
    image: &mut RgbaImage,
    most_present_color: RgbColor,
    max_distance: f32,
) {
    for pixel in image.pixels_mut() {
        let Rgba([r, g, b, _a]) = *pixel;
        let rgb = (r, g, b);
        let distance = most_present_color.color_distance(rgb);

        if distance < max_distance {
            let new_alpha = map_distance_to_transparency(distance, max_distance);
            *pixel = Rgba([r, g, b, new_alpha]);
        }
    }
}

fn _apply_gaussian_blur_to_alpha(image: &mut RgbaImage, sigma: f32) {
    let (width, height) = image.dimensions();
    let mut alpha_image = RgbaImage::new(width, height);

    // Extract the alpha channel
    for (x, y, pixel) in image.enumerate_pixels() {
        alpha_image.put_pixel(x, y, Rgba([0, 0, 0, pixel[3]]));
    }

    // Apply Gaussian blur to the alpha channel
    let blurred_alpha_image = gaussian_blur_f32(&alpha_image, sigma);

    // Update the image with the blurred alpha channel
    for (x, y, blurred_pixel) in blurred_alpha_image.enumerate_pixels() {
        let pixel = image.get_pixel_mut(x, y);
        pixel[3] = blurred_pixel[3];
    }
}

#[derive(Clone, Debug)]
pub struct ImageInformation {
    pub brightness: Brightness,
    pub grayscale_similarity: GrayScaleSimilarity,
    pub color_map: ColorMap,
    pub image_type: Option<ImageType>,
}

impl ImageInformation {
    pub fn new() -> Self {
        ImageInformation {
            brightness: Brightness {
                average: 0.0,
                min: 0.0,
                max: 0.0,
            },
            grayscale_similarity: GrayScaleSimilarity {
                average: 0.0,
                min: 0.0,
                max: 0.0,
            },
            color_map: ColorMap {
                most_present_color: (0, 0, 0),
                most_present_color_percentage: 0.0,
                amount: 0,
            },
            image_type: None,
        }
    }
}
#[derive(Clone, Debug)]
pub struct GrayScaleSimilarity {
    pub average: f32,
    pub min: f32,
    pub max: f32,
}
#[derive(Clone, Debug)]
pub struct ColorMap {
    pub most_present_color: (u8, u8, u8),
    pub most_present_color_percentage: f64,
    pub amount: u64,
}
#[derive(Clone, Debug)]
pub struct Brightness {
    pub average: f32,
    pub min: f32,
    pub max: f32,
}

fn get_image_information(image: &RgbaImage) -> ImageInformation {
    let mut total_brightness = 0.0;
    let mut total_grayscale = 0.0;
    let mut color_map: HashMap<(u8, u8, u8), u64> = HashMap::new();
    let mut image_information = ImageInformation::new();
    let mut min_brightness = f32::MAX;
    let mut max_brightness = f32::MIN;
    let mut min_grayscale = f32::MAX;
    let mut max_grayscale = f32::MIN;

    let num_pixels = image.width() * image.height();
    const SAMPLE_DISTANCE: usize = 50;
    let pixel_amount = num_pixels / SAMPLE_DISTANCE.max(1) as u32;

    for (i, Rgba([r, g, b, a])) in image.pixels().enumerate() {
        if i % SAMPLE_DISTANCE != 0 || *a <= 128 {
            continue;
        }
        let pixel = RgbColor {
            r: *r,
            g: *g,
            b: *b,
        };
        let brightness = pixel.brightness();
        let grayscale_similarity = pixel.calculate_grayscale_similarity();

        total_brightness += brightness;
        total_grayscale += grayscale_similarity;

        if brightness < min_brightness {
            min_brightness = brightness;
        }
        if brightness > max_brightness {
            max_brightness = brightness;
        }
        if grayscale_similarity < min_grayscale {
            min_grayscale = grayscale_similarity;
        }
        if grayscale_similarity > max_grayscale {
            max_grayscale = grayscale_similarity;
        }

        *color_map.entry((pixel.r, pixel.g, pixel.b)).or_insert(0) += 1;
    }

    let average_brightness = total_brightness / pixel_amount as f32;
    let average_grayscale_similarity = total_grayscale / pixel_amount as f32;

    let (most_present_color, &most_present_color_count) = color_map
        .iter()
        .max_by_key(|&(_, count)| count)
        .unwrap_or((&(0, 0, 0), &0));
    let most_present_color_percentage = most_present_color_count as f64 / pixel_amount as f64;
    let color_amount = color_map.len() as u64;

    image_information.brightness = Brightness {
        average: average_brightness,
        min: min_brightness,
        max: max_brightness,
    };

    image_information.grayscale_similarity = GrayScaleSimilarity {
        average: average_grayscale_similarity,
        min: min_grayscale,
        max: max_grayscale,
    };

    image_information.color_map = ColorMap {
        most_present_color: *most_present_color,
        most_present_color_percentage,
        amount: color_amount,
    };
    // predict image type
    if image_information.color_map.amount < 2000 // avg < 500
        && image_information.color_map.most_present_color_percentage > 0.1
    // mostly white
    {
        image_information.image_type = Some(ImageType::Cartoon);
    } else {
        image_information.image_type = Some(ImageType::Picture);
    }
    image_information
}

fn preprocess_image(image: &DynamicImage, options: &NordOptions) -> Array4<f32> {
    let model = options.model.to_struct();
    let nwidth: u32 = model.width;
    let nheight: u32 = model.height;
    let resized = image.resize_exact(nwidth, nheight, image::imageops::FilterType::Nearest);
    let rgb_image = resized.to_rgb8();

    let mut input_tensor = Array4::<f32>::zeros((1, 3, nheight as usize, nwidth as usize));
    for (y, x, pixel) in rgb_image.enumerate_pixels() {
        input_tensor[[0, 0, x as usize, y as usize]] = pixel[0] as f32 / 255.0;
        input_tensor[[0, 1, x as usize, y as usize]] = pixel[1] as f32 / 255.0;
        input_tensor[[0, 2, x as usize, y as usize]] = pixel[2] as f32 / 255.0;
    }

    input_tensor
}

fn segment_image<'a>(
    session: &'a mut Session<'_>,
    image: &DynamicImage,
    options: &NordOptions,
) -> Result<
    onnxruntime::tensor::OrtOwnedTensor<'a, 'a, f32, ndarray::Dim<ndarray::IxDynImpl>>,
    Box<dyn std::error::Error>,
> {
    let input_tensor = preprocess_image(image, &options);
    println!("Input tensor shape: {:?}", input_tensor.shape());
    let input_array = vec![input_tensor];
    let output: Vec<OrtOwnedTensor<f32, ndarray::Dim<ndarray::IxDynImpl>>> =
        session.run(input_array).unwrap();
    println!("Output tensor shape: {:?}", output[0].shape());
    let tensor = output.into_iter().next().unwrap();
    Ok(tensor)
}

fn apply_mask(
    image: &DynamicImage,
    mask: &onnxruntime::tensor::OrtOwnedTensor<f32, ndarray::Dim<ndarray::IxDynImpl>>,
    options: &NordOptions,
) -> DynamicImage {
    let (orig_width, orig_height) = image.dimensions();
    let mask_width = mask.shape()[2] as u32;
    let mask_height = mask.shape()[3] as u32;

    // Convert the mask to a Vec<u8> by scaling f32 values to u8
    let mask_data: Vec<u8> = mask
        .to_slice()
        .unwrap()
        .iter()
        .map(|&v| (v * 255.0).min(255.0).max(0.0) as u8)
        .collect();

    // Ensure mask dimensions match image dimensions
    let resized_mask = DynamicImage::ImageLuma8(
        image::GrayImage::from_raw(mask_width, mask_height, mask_data).unwrap(),
    )
    .resize_exact(
        orig_width,
        orig_height,
        image::imageops::FilterType::Nearest,
    )
    .to_luma8();

    let mut masked_image = RgbaImage::new(orig_width, orig_height);

    // choose activation function
    let mut activation_function: fn(u8) -> u8 = |x| x;
    let sigmoid = |x: u8| -> u8 {
        if x < 5 {
            return 0;
        } else if x > 250 {
            return 255;
        }
        let x = x as f32 / 255.0; // Normalize to range [0, 1]
        let sigmoid_value = 255.0 * ((1.0) / (1.0 + (((x - 0.5) / -0.1).exp())));
        sigmoid_value as u8
    };

    if options.activation_function == ActivationFunction::Sigmoid {
        activation_function = sigmoid;
    }
    // time start
    let start = std::time::Instant::now();
    masked_image
        .chunks_exact_mut(4)
        .enumerate()
        .for_each(|(index, pixel)| {
            let x = (index as u32) % orig_width;
            let y = (index as u32) / orig_width;
            let pixel_value = image.get_pixel(x, y);
            let mask_value = resized_mask.get_pixel(x, y)[0];

            let [r, g, b, a] = pixel_value.0;
            if a < mask_value {
                pixel.copy_from_slice(&[r, g, b, a]);
                return;
            }
            let alpha = activation_function(mask_value);
            pixel.copy_from_slice(&[r, g, b, alpha]);
        });
    println!(
        "[Masking-loop] Time taken: {:.3} seconds",
        start.elapsed().as_secs_f32()
    );
    let img = DynamicImage::ImageRgba8(masked_image);
    img
}

pub fn remove_background<'a>(
    mut session: Session<'_>,
    image: DynamicImage,
    options: &NordOptions,
) -> DynamicImage {
    // start time
    let start = std::time::Instant::now();
    // generates black-white mask
    let mask = segment_image(&mut session, &image, &options).unwrap();
    println!(
        "[Segmentation] Time taken: {:.3} seconds",
        start.elapsed().as_secs_f32()
    );
    let start = std::time::Instant::now();
    // apply mask to image
    let segmented_image = apply_mask(&image, &mask, &options);
    println!(
        "[Masking] Time taken: {:.3} seconds",
        start.elapsed().as_secs_f32()
    );
    segmented_image
}
