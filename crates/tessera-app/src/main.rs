// Hide the console window that Windows would otherwise open behind the app.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    tessera_app_lib::run();
}
