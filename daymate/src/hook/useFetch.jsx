// hook/useFetch.jsx
import { useState, useEffect, useCallback } from "react";

export default function useFetch(url, params = {}, autoFetch = true) {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(autoFetch);
    const [error, setError] = useState(null);

    const fetchData = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            // Build query string from params
            const query = new URLSearchParams(params).toString();
            const fullUrl = query ? `${url}?${query}` : url;

            const res = await fetch(fullUrl);
            if (!res.ok) throw new Error(`Fetch failed: ${res.status}`);
            const json = await res.json();
            setData(json);
        } catch (err) {
            setError(err.message);
            setData(null);
        } finally {
            setLoading(false);
        }
    }, [url, JSON.stringify(params)]); // Re-run if url or params change

    // Auto fetch on mount if enabled
    useEffect(() => {
        if (autoFetch) fetchData();
    }, [fetchData, autoFetch]);

    return { data, loading, error, fetchData };
}


/*HOW TO USE in a component:
// Weather fetch (auto-fetch only when lat/lon available)
  const { data: weather, loading: weatherLoading, fetchData: fetchWeather } = useFetch(
    `${BASE_URL}/weather`,
    { lat: location.lat, lon: location.lon },
    false // don't auto fetch until lat/lon is ready
  );

  // News fetch (auto-fetch on mount)
  const [newsCountry, setNewsCountry] = useState("bd");
  const { data: news, loading: newsLoading, fetchData: fetchNews } = useFetch(
    `${BASE_URL}/news`,
    { country_name: newsCountry }
  );

  // Trigger weather fetch when location updates
  useEffect(() => {
    if (location.lat && location.lon) fetchWeather();
  }, [location.lat, location.lon, fetchWeather]);

* */