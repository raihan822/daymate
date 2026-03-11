import useFetch from "./useFetch.jsx";
import {BASE_URL} from "../api/baseUrl.js";

export default function useNews(country = "bd"){

    const {
        data: news,
        loading: isNewsLoading,
        fetchData: fetchNews
    } = useFetch(
        `${BASE_URL}/news`,
        {params:{country_name: country}},
        true
    );
    const infoMessage =
        news?.information?.realTimeArticles?.message ??
        news?.articlesRemovedFromResponse?.historicalArticles?.message ??
        null;

    const headlines = news?.articles?.slice(0, 10) ?? [];
    //<python> headlines = [a.get("title") for a in news.get("articles", [])[:10]]  # Safe extraction of the dict.get() value with default value []
    // const headlines = news?.articles?.slice(0, 10).map(a => a.title) || [];

    return {
        news,
        headlines,
        isNewsLoading,
        infoMessage,
        fetchNews
    };
}