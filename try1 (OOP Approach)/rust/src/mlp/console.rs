use std::io::{Write, stdout};

pub fn update_progress_bar(current: usize, total: usize) {
    // Prevent division by zero
    if total == 0 {
        return;
    }

    // Calculate progress percentage
    let progress = current as f32 / total as f32;

    // Get terminal width (with fallback)
    let width = match term_size::dimensions() {
        Some((w, _)) => w as usize,
        None => 80, // Default fallback width
    };

    // Leave room for percentage display and brackets
    let bar_width = width.saturating_sub(10);
    let filled_width = (progress * bar_width as f32) as usize;

    // Build the progress bar
    let mut bar = String::with_capacity(width);
    bar.push('[');
    for i in 0..bar_width {
        if i < filled_width {
            bar.push('=');
        } else if i == filled_width && current < total {
            bar.push('>');
        } else {
            bar.push(' ');
        }
    }
    bar.push(']');

    // Add percentage
    let percentage = progress * 100.0;
    bar.push_str(&format!(" {:3.0}%", percentage));

    // Print the bar, overwriting previous line
    print!("\r{}", bar);
    // Ignore any potential flush errors
    let _ = stdout().flush();
}
