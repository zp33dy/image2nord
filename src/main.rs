#![warn(clippy::str_to_string)]
mod commands;
use colors::{ImageInformation, NordOptions, RgbColor};
use config::Config;
use poise::serenity_prelude as serenity;
use dotenv::dotenv;
use ::serenity::all::{
    Attachment, ButtonStyle, CacheHttp, ComponentInteraction, CreateAttachment, CreateButton, CreateInteractionResponse, CreateInteractionResponseFollowup, CreateInteractionResponseMessage, CreateMessage, CreateQuickModal, EditAttachments, EditInteractionResponse, InputTextStyle, Interaction, Message, ModalInteraction, QuickModalResponse, ReactionType
};
use std::{
    env, io::Cursor, sync::{Arc, Mutex}, time::Duration
};
use anyhow::{bail, Result};
use reqwest;
use image::DynamicImage;

// Types used by all command functions
type AsyncError = Box<dyn std::error::Error + Send + Sync>;
type Context<'a> = poise::Context<'a, Data, AsyncError>;
type SContext = serenity::Context;
use log::{info, warn};
use ttl_cache::TtlCache;
use tokio::sync::RwLock;
use lru_time_cache::LruCache;
use bytes::Bytes;

mod config;
mod colors;
mod tickbox;
mod visual_scale;
mod brightnes_image;
mod db;

// Custom user data passed to all command functions


struct ImageCache {
    cache: RwLock<LruCache<String, (DynamicImage, ImageInformation)>>,
}

impl ImageCache {
    fn new(capacity: usize, ttl: Duration) -> Self {
        Self {
            cache: RwLock::new(LruCache::with_expiry_duration_and_capacity(ttl, capacity)),
        }
    }

    async fn insert(&self, key: String, image: DynamicImage, info: ImageInformation) {
        let mut cache = self.cache.write();
        cache.await.insert(key, (image, info));
    }

    async fn get(&self, key: &str) -> Option<(DynamicImage, ImageInformation)> {
        let mut cache = self.cache.write();
        cache.await.get(key).map(|(image, info)| {
            (image.clone(), info.clone())
        })
    }
}
pub struct Data {
    image_cache: ImageCache,
    config: Config,
    
}

async fn on_error(error: poise::FrameworkError<'_, Data, AsyncError>) {
    // This is our custom error handler
    // They are many errors that can occur, so we only handle the ones we want to customize
    // and forward the rest to the default handler
    match error {
        poise::FrameworkError::Setup { error, .. } => panic!("Failed to start bot: {:?}", error),
        poise::FrameworkError::Command { error, ctx, .. } => {
            println!("Error in command `{}`: {:?}", ctx.command().name, error,);
        }
        error => {
            if let Err(e) = poise::builtins::on_error(error).await {
                println!("Error while handling error: {}", e)
            }
        }
    }
}

async fn interaction_create(ctx: SContext, interaction: Interaction, data: &Data) -> Option<()> {
    if let Interaction::Component(interaction) = interaction {
        let content = &interaction.data.custom_id;
        if content.starts_with("darken-") {
            handle_interaction_darkening(&ctx, &interaction, data).await.unwrap();
        }
        if content.starts_with("delete-") {
            let message_id = content.split("-").last().unwrap().parse::<u64>().unwrap();
            handle_dispose(&ctx, &interaction, message_id).await.unwrap();
        }
        if content.starts_with("clear-") {
            initial_clear_components(&ctx, &interaction).await.unwrap();
        }
        if content.starts_with("stop-") {
            let response = CreateInteractionResponse::UpdateMessage(CreateInteractionResponseMessage::default());
            interaction.create_response(&ctx, response).await.unwrap();
            interaction.delete_response(&ctx).await.unwrap();
        }
    }
    Some(())
}

async fn fetch_or_raise_message(
    ctx: &SContext, 
    interaction: &ComponentInteraction, 
    message_id: u64
) -> Message {
    let message = interaction.channel_id.message(&ctx, message_id).await;
    if message.is_err() {
        // fetch image message -> error
        let response = CreateInteractionResponse::Message(CreateInteractionResponseMessage::new()
            .content("Seems like the bright picture has vanished. I can't darken what I can't see.")
        );
        interaction.create_response(&ctx, response).await.unwrap();
    }
    message.unwrap()
}
// Returns the color as hex, or err
async fn modal_get_color(ctx: &SContext, interaction: &ComponentInteraction) -> Result<(RgbColor, ModalInteraction)> {
    let modal = CreateQuickModal::new("Enter a Color")
        .timeout(std::time::Duration::from_secs(600))
        .short_field("Color (hex) e.g. #AF4453");
    let response = interaction.quick_modal(ctx, modal).await?;
    let response = response.unwrap(); 
    let (color_code) = (&response.inputs[0]);
    let color = match RgbColor::from_hex(&color_code) {
        Ok(color) => {
            (color, response.interaction)
        },
        Err(e) => {
            response
                .interaction
                .create_response(ctx, (|| {
                    CreateInteractionResponse::Message(CreateInteractionResponseMessage::new()
                        .content(format!("Invalid color code: {}", e))
                    )
                })()).await?;
            bail!("Invalid color code: {}", e);
        }
    };
    Ok(color)
}

enum AnyInteraction {
    Component(ComponentInteraction),
    Modal(ModalInteraction),
}

impl AnyInteraction {
    async fn create_response(&self, cache_http: impl CacheHttp, builder: CreateInteractionResponse) -> Result<(), serenity::Error>
    {
        match self {
            AnyInteraction::Component(interaction) => {
                interaction.create_response(cache_http, builder).await
            },
            AnyInteraction::Modal(interaction) => {
                interaction.create_response(cache_http, builder).await
            },
        }
    }

    async fn edit_response(&self, cache_http: impl CacheHttp, builder: EditInteractionResponse) -> Result<Message, serenity::Error>
    {
        match self {
            AnyInteraction::Component(interaction) => {
                interaction.edit_response(cache_http, builder).await
            },
            AnyInteraction::Modal(interaction) => {
                interaction.edit_response(cache_http, builder).await
            },
        }
    }

    fn custom_id(&self) -> &str {
        match self {
            AnyInteraction::Component(interaction) => &interaction.data.custom_id,
            AnyInteraction::Modal(interaction) => &interaction.data.custom_id,
        }
    }
}

async fn handle_interaction_darkening(ctx: &SContext, interaction: &ComponentInteraction, data: &Data) -> Result<()> {
    let content = &interaction.data.custom_id;
    let mut options = NordOptions::from_custom_id(&content);
    let message_id = content.split("-").last().unwrap().parse::<u64>()?;
    let _update = content.split("-").nth(1).unwrap().parse::<bool>().unwrap_or(true);

    let mut current_interaction = AnyInteraction::Component(interaction.clone());

    // ask for background color
    if options.background_color == Some(RgbColor::from_hex("000001").unwrap()) {
        let color: RgbColor;
        let new_interaction: ModalInteraction;
        (color, new_interaction) = match modal_get_color(&ctx, &interaction).await {
            Ok(color) => color,
            Err(_) => {
                // Error handled inside modal_get_color
                return Ok(());
            }
        };
        options.background_color = Some(color);
        current_interaction = AnyInteraction::Modal(new_interaction);
    }
    let mut message: Option<Message> = None;

    // auto adjust options
    if options.auto_adjust {
        message = Some(fetch_or_raise_message(&ctx, &interaction, message_id).await);
        let ref unwrapped = message.as_ref().unwrap();
        let (_image, information) = fetch_image(unwrapped.attachments.first().unwrap(), data).await;
        let new_options = NordOptions::from_image_information(&information);
        options = NordOptions {start: options.start, ..new_options};
    }

    let new_components = options.build_componets(message_id, true);
    
    println!("options: {:?}", options);

    if options.start {
        // start button pressed
        if message.is_none() {
            message = Some(fetch_or_raise_message(&ctx, &interaction, message_id).await);
        }
        let response = CreateInteractionResponse::Acknowledge;
        current_interaction.create_response(&ctx, response).await?;
        // edit response with new components
        let response = EditInteractionResponse::new()
            .attachments(EditAttachments::keep_all(&interaction.message))
            .content("⌛ I'm working on it. Please wait a moment.")
            .components(new_components.clone());
        current_interaction.edit_response(&ctx, response).await.unwrap();
    } else {
        // first ack, that existing image is being kept
        let response = CreateInteractionResponse::Acknowledge;
        current_interaction.create_response(&ctx, response).await?;
        // edit response with new components
        let response = EditInteractionResponse::new()
            .attachments(EditAttachments::keep_all(&interaction.message))
            .content("⌛ I change the options. Please wait a moment.")
            .components(new_components.clone());
        current_interaction.edit_response(&ctx, response).await.unwrap();
    }
    
    if !options.start {
        let response = EditInteractionResponse::new()
            .content("Edited your options.")
            .components(new_components.clone())
        ;
        current_interaction.edit_response(&ctx, response).await?;
        return Ok(())
    }
    // ensure existence of message
    if message.is_none() {
        current_interaction.edit_response(&ctx, EditInteractionResponse::new()
            .content("Seems like the bright picture has vanished. I can't darken what I can't see.")
        ).await?;
        return Ok(())
    }
    let message = message.unwrap();
    // process image
    let buffer = match process_attachments(&message, &data, &options).await {
        Ok(buffer) => buffer,
        Err(e) => {
            current_interaction.edit_response(&ctx, EditInteractionResponse::default().content(e.to_string())).await?;
            return Ok(())
        }
    };
    let attachment = CreateAttachment::bytes(buffer, "image.webp");
    let content = EditInteractionResponse::new()
        .new_attachment(attachment)
        .content("Here it is! May I delete your shiny one?")
        .components(new_components.clone())
    ;
    // stone emoji: 
    println!("sending message");
    current_interaction.edit_response(&ctx, content).await?;
    Ok(())
}

pub async fn process_attachments(message: &Message, data: &Data, options: &NordOptions) -> Result<Vec<u8>, AsyncError>{
    for attachment in &message.attachments {
        println!("Processing attachment");
        let image = process_image(&attachment, data, options.clone()).await.unwrap();
        println!("writing image to buffer");
        let mut buffer = Vec::new();
            // Create a PNG encoder with a specific compression level
        {
            let mut cursor = Cursor::new(&mut buffer);
            //let encoder = PngEncoder::new_with_quality(&mut cursor, CompressionType::Fast, FilterType::Adaptive);
            image.write_to( &mut cursor, image::ImageFormat::WebP).expect("Failed to write image to buffer");
            //encoder.write_image(&image.as_bytes(), image.width(), image.height(), image::ExtendedColorType::Rgba8).unwrap();
        }
        return Ok(buffer);
    }
    panic!("No attachment found in message");
}
async fn initial_clear_components(ctx: &SContext, interaction: &ComponentInteraction) -> Result<()> {
    // fetch message
    let response = CreateInteractionResponse::Acknowledge;
    interaction.create_response(&ctx, response).await?;
    let response = EditInteractionResponse::new()
        .attachments(EditAttachments::keep_all(&interaction.message))
        .content("")
        .components(vec![]);
    interaction.edit_response(&ctx, response).await?;
    Ok(())
}

async fn handle_dispose(ctx: &SContext, interaction: &ComponentInteraction, message_id: u64) -> Result<()> {
    initial_clear_components(&ctx, &interaction).await?;
    // fetch message
    interaction.channel_id.delete_message(&ctx, message_id).await?;
    let response =
        CreateInteractionResponseFollowup::new()
        .content("I have thrown it deep into the void to never see it again. Enjoy the darkness!")
        .ephemeral(true)
    ;
    interaction.create_followup(&ctx, response).await?;
    Ok(())
}


#[tokio::main]
async fn main() {
    // env_logger::init();
    dotenv().ok();
    // FrameworkOptions contains all of poise's configuration option in one struct
    // Every option can be omitted to use its default value
    let image_cache = ImageCache::new(20, Duration::from_secs(600));
    let options = poise::FrameworkOptions {
        commands: vec![commands::edit_message_image(), commands::help()],
        prefix_options: poise::PrefixFrameworkOptions {
            prefix: Some("~".into()),
            edit_tracker: Some(Arc::new(poise::EditTracker::for_timespan(
                Duration::from_secs(3600),
            ))),
            additional_prefixes: vec![
                poise::Prefix::Literal("nanachi"),
                poise::Prefix::Literal("nanachi,"),
            ],
            ..Default::default()
        },
        // The global error handler for all error cases that may occur
        on_error: |error| Box::pin(on_error(error)),
        // This code is run before every command
        pre_command: |ctx| {
            Box::pin(async move {
                println!("Executing command {}...", ctx.command().qualified_name);
            })
        },
        // This code is run after a command if it was successful (returned Ok)
        post_command: |ctx| {
            Box::pin(async move {
                println!("Executed command {}!", ctx.command().qualified_name);
            })
        },
        // Every command invocation must pass this check to continue execution
        // command_check: Some(|ctx| {
        //     Box::pin(async move {
        //         if ctx.author().id == 123456789 {
        //             return Ok(false);
        //         }
        //         Ok(true)
        //     })
        // }),
        // Enforce command checks even for owners (enforced by default)
        // Set to true to bypass checks, which is useful for testing
        skip_checks_for_owners: false,
        event_handler: |ctx, event, framework, data| {
            Box::pin(event_handler(ctx, event, framework, data))
        },
        ..Default::default()
    };

    let framework = poise::Framework::builder()
        .setup(move |ctx, _ready, framework| {
            Box::pin(async move {
                println!("Logged in as {}", _ready.user.name);
                poise::builtins::register_globally(ctx, &framework.options().commands).await?;
                Ok(Data {
                    image_cache: image_cache,
                    config: config::load_config(),
                })
            })
        })
        .options(options)
        .build();

    for (key, value) in env::vars() {
        println!("{}: {}", key, value);
    }
    // load DISCORD_TOKEN from .env file
    let token = env::var("DISCORD_TOKEN").expect("DISCORD_TOKEN must be set in .env");
    let intents =
        serenity::GatewayIntents::non_privileged() | serenity::GatewayIntents::MESSAGE_CONTENT;

    let client = serenity::ClientBuilder::new(token, intents)
        .framework(framework)
        .await;

    client.unwrap().start().await.unwrap()
}


async fn event_handler(
    ctx: &SContext,
    event: &serenity::FullEvent,
    _framework: poise::FrameworkContext<'_, Data, AsyncError>,
    data: &Data,
) -> Result<(), AsyncError> {
    println!(
        "Got an event in event handler: {:?}",
        event.snake_case_name()
    );

    match event {
        serenity::FullEvent::Ready { data_about_bot, .. } => {
            println!("Logged in as {}", data_about_bot.user.name);
        }
        serenity::FullEvent::InteractionCreate { interaction, .. } => {
            interaction_create(ctx.clone(), interaction.clone(), data).await;
        }
        serenity::FullEvent::Message { new_message: message } => {
            for attachment in &message.attachments {
                if message.author.bot {
                    continue;
                }
                println!("attachment found");
                println!(
                    "media type: {:?}; filename: {}; Size: {} MiB; URL: {}", 
                    attachment.content_type, attachment.filename, attachment.size as f64 / 1024.0 / 1024.0, attachment.url
                );
                ask_user_to_darken_image(&ctx, &message, &attachment, data).await?;
            }
        }
        _ => {}
    }
    Ok(())
}


async fn image_check(attachment: &Attachment) -> Result<()> {
    let mib = attachment.size as f64 / 1024.0 / 1024.0;
    if mib > 16.0 {
        bail!("File too large: {} MiB", mib);
    }
    if attachment.content_type.is_none() {
        bail!("No content type found for attachment");
    }
    let content_type = attachment.content_type.as_ref().unwrap();
    if !content_type.starts_with("image/") {
        bail!("Attachment is not an image: {}", content_type);
    }
    Ok(())
}


pub async fn fetch_image_and_info(attachment: &Attachment, data: &Data) -> Result<(DynamicImage, ImageInformation)> {
    image_check(attachment).await?;
    let url = attachment.url.clone();
    let image_and_info = {
        let image = data.image_cache.get(&url).await;
        if image.is_none() {
            let image = download_image(&attachment).await?;
            Ok::<(DynamicImage, ImageInformation), anyhow::Error>(
                (image.clone(), colors::calculate_average_brightness(&image.to_rgba8()))
            )
        } else {
            Ok(image.unwrap())
        }
    };
    image_and_info
}


async fn ask_user_to_darken_image(
    ctx: &SContext, 
    message: &Message, 
    attachment: &Attachment, 
    data: &Data
) -> Result<(), anyhow::Error> {

    // download image or get from cache
    image_check(attachment).await?;
    let url = attachment.url.clone();
    let image_and_info = fetch_image_and_info(attachment, data).await;
    let (image, info) = image_and_info
        .expect("Image or info is none in ask_user_to_darken_image");
    println!("inserting");
    data.image_cache.insert(url, image.clone(), info.clone()).await;
    let bright = info.brightness.average;
    if bright < data.config.threshold.brightness {
        panic!("Not bright enough: {bright}")
    }
    
    let start = std::time::Instant::now();
    let image_scale = brightnes_image::generate_image(bright, 1.0, 9.0);
    let mut buffer = Cursor::new(Vec::new()); // Use Cursor to add Seek capability
    println!("Pre save {:?}", start.elapsed());
    image_scale.write_to(&mut buffer, image::ImageFormat::WebP).expect("Failed to write image to buffer");
    println!("After save {:?}", start.elapsed());
    // Optionally, reset cursor position to the beginning if you need to read from it afterward
    buffer.set_position(0);
    let attachment = CreateAttachment::bytes(buffer.into_inner(), "scale.webp");

    println!("Generated image in {:?}", start.elapsed());
    let response = CreateMessage::new()
        .content(
            format!(
                "Bruhh... This looks bright as fuck. On a scale **from 1 to 9 it's a {:.1}**.\nMay I darken it?", 
                bright*8. + 1.)
        )
        .files(vec![attachment])
        .button(CreateButton::new(
            NordOptions::new().make_nord_custom_id(&message.id.into(), false, None)
        )
            .style(ButtonStyle::Primary)
            .emoji("🌙".parse::<ReactionType>().unwrap())
        )
        .button(CreateButton::new(format!("stop-{}", message.id))
            .style(ButtonStyle::Primary)
            .label("No")
        );
    let _new_message = message.channel_id.send_message(&ctx, response).await?;
    Ok(())
}


async fn fetch_image(
    attachment: &serenity::Attachment, 
    data: &Data, 
) -> (DynamicImage, ImageInformation) {
    image_check(attachment).await.unwrap();
    let url = attachment.url.clone();
    let image_and_info: std::result::Result<(DynamicImage, ImageInformation), anyhow::Error> = {
        let image = data.image_cache.get(&url).await;
        if image.is_none() {
            let image = download_image(&attachment).await.unwrap();
            Ok::<(DynamicImage, ImageInformation), anyhow::Error>(
                (image.clone(), colors::calculate_average_brightness(&image.to_rgba8()))
            )
        } else {
            Ok(image.unwrap())
        }
    };
    image_and_info.unwrap()
}


async fn process_image(attachment: &serenity::Attachment, data: &Data, options: colors::NordOptions) -> Result<DynamicImage> {
    let (image, info) = fetch_image(attachment, data).await;
    Ok(colors::apply_nord(image, options, &info))
}

async fn download_image(attachment: &Attachment) -> Result<DynamicImage> {
    // Send the GET request
    //println!("Downloading: {}=&format=png", attachment.proxy_url);
    let response = reqwest::get(format!("{}=&format=png", attachment.proxy_url)).await?;
    
    // Ensure the request was successful
    if !response.status().is_success() {
        info!("Request failed with status code: {}", response.status());
        anyhow::bail!("Request failed with status code: {}", response.status());
    }
   
    let bytes = response.bytes().await?;
    // let raw = attachment.download().await?;
    // Get the image bytes
    println!("Downloaded image with {} bytes", bytes.len());
    // Load the image from the bytes
    let image = image::load_from_memory(&bytes).map_err(
        |e| anyhow::anyhow!("Failed to load image: {}", e)
    )?;
    
    Ok(image)
}