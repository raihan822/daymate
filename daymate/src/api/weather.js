import { BASE_URL } from "./baseUrl.js";
//
// export async function getWeather(lat, lon) {
//     if (!lat || !lon) return null;
//     const res = await fetch(`${BASE_URL}/weather?lat=${lat}&lon=${lon}`);
//     if (!res.ok) throw new Error("Failed to fetch weather");
//     return res.json();
// }