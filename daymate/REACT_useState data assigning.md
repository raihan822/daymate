- React always executes `Return ()=>{}` arrow func or `onClick ()=>{}` by itself when required
* WHY REACT CALLS this portion by itself and I dont have to call the function manually?
        react renders only the return portion itself always! So, react k diye future e koraite chaile ba onClick js e kichu korte chaile 
        ALWAYS USE ARRO FUNCTION ()=>{..}



# Question:
```
export default function ProductList() { 
    const [products, setProducts] = useState([]); 
    const [loading, setLoading] = useState(true); 
    const [error, setError] = useState(null); 
    const URL = "https://crud.teamrabbil.com/api/v1/ReadProduct";
    useEffect(
        () => {}, 
        []
    );
    //..........
}
```
What is the best practice, will I keep the const variable names of the useState var products and etc lowercase? and will it be [] or what type is best practice when api call? And how do we assign value into these useState variables with set in different types [],{},"" etc etc?


# Answer:
Good questions — these are **core React best-practices**, and you’re already very close to doing things the *right way*. I’ll answer point-by-point with **clear rules + reasoning**, not vague advice.

---

## 1️⃣ Naming of `useState` variables

### Should they be lowercase?

**Yes. Always use `camelCase` (lowercase first letter).**

```js
const [products, setProducts] = useState([]);
const [loading, setLoading] = useState(true);
const [error, setError] = useState(null);
```

### Why?

* React convention
* JavaScript variable standard
* Uppercase is reserved for **Components** (`ProductList`, `TableComponent`)

❌ Bad:

```js
const [Products, SetProducts] = useState([]);
```

✅ Good:

```js
const [products, setProducts] = useState([]);
```

**Rule:**

> Variables & state → camelCase
> Components → PascalCase

---

## 2️⃣ What should be the initial value of `useState` for API data?

### Best practice rule:

> **Initial state must match the expected final data type**

### Your API:

```js
response.data.data   // ← array of products
```

So this is **correct**:

```js
const [products, setProducts] = useState([]);
```

### Why not `{}` or `null`?

Because later you’ll do things like:

```js
products.map(...)
```

If `products` were `{}` or `null`, your app would crash.

---

### Common patterns (VERY IMPORTANT)

| API Response Type | Initial State     |
| ----------------- | ----------------- |
| Array (`[]`)      | `useState([])`    |
| Object (`{}`)     | `useState({})`    |
| String            | `useState("")`    |
| Number            | `useState(0)`     |
| Boolean           | `useState(false)` |
| Unknown / async   | `useState(null)`  |

---

## 3️⃣ How do we assign values using `setState` correctly?

### 🔹 Array state

```js
const [products, setProducts] = useState([]);
setProducts(response.data.data);
```

Appending:

```js
setProducts(prev => [...prev, newProduct]);
```

---

### 🔹 Object state

```js
const [user, setUser] = useState({});
setUser(response.data);
```

Updating one property:

```js
setUser(prev => ({
    ...prev,
    name: "Raihan"
}));
```

---

### 🔹 String state

```js
const [search, setSearch] = useState("");
setSearch("laptop");
```

---

### 🔹 Boolean state

```js
const [loading, setLoading] = useState(true);
setLoading(false);
```

Toggle:

```js
setLoading(prev => !prev);
```

---

### 🔹 Nullable data (VERY common in APIs)

```js
const [selectedProduct, setSelectedProduct] = useState(null);
```

Later:

```js
setSelectedProduct(product);
```

Rendering safely:

```js
{selectedProduct && <ProductDetails />}
```

---

## 4️⃣ Best-practice structure for API state (YOU DID IT RIGHT ✅)

```js
const [data, setData] = useState([]);
const [loading, setLoading] = useState(true);
const [error, setError] = useState(null);
```

This is **industry-standard** and used in:

* React docs
* Production apps
* Interviews

---

## 5️⃣ Your current code — verdict

### ✅ You followed best practice

✔ Correct naming
✔ Correct initial types
✔ Correct `useEffect` usage
✔ Correct API handling

Only **tiny improvement** 👇

### Optional improvement (cleaner)

Move URL outside component:

```js
const URL = "https://crud.teamrabbil.com/api/v1/ReadProduct";

export default function ProductList() {
    ...
}
```

---

## 6️⃣ Interview-ready one-liner answers (memorize)

> **Q:** Why use `useState([])` for API calls?
> **A:** Because the API returns an array and React state should match the expected data type to avoid runtime errors.

> **Q:** Why camelCase for state?
> **A:** JavaScript convention — uppercase is reserved for components.

---

If you want, next I can show you:

* ✅ **Type-safe pattern using PropTypes**
* ✅ **How to handle empty API response**
* ✅ **Best structure for large apps (service layer)**
* ✅ **React Query / SWR (real-world approach)**

Just tell me 👍

