"""Test script for hex swatch inline rendering in Streamlit markdown."""
import streamlit as st

st.set_page_config(page_title="Hex Swatch Test", page_icon="🎨")

st.title("🎨 Hex Swatch Test Suite")

st.header("Basic Hex Swatches")
st.markdown("Single color: `#FF5733`")
st.markdown("Multiple colors: `#FF0000` `#00FF00` `#0000FF`")

st.header("Primary Colors")
st.markdown("""
- Red: `#FF0000`
- Green: `#00FF00`
- Blue: `#0000FF`
""")

st.header("Custom Colors")
st.markdown("""
Here are some custom colors:
- Primary: `#FF5733`
- Secondary: `#33FF57`
- Accent: `#3357FF`
- Custom: `#A1B2C3`
""")

st.header("Mixed Case")
st.markdown("Lowercase: `#abc123`")
st.markdown("Uppercase: `#ABC123`")
st.markdown("Mixed: `#AbC123`")

st.header("In Sentences")
st.markdown("The primary brand color is `#FF5733` and the secondary is `#00FF00`.")
st.markdown("Our theme uses `#0000FF` for links and `#FFA500` for warnings.")

st.header("Edge Cases - Should NOT Render")
st.markdown("Missing backticks: #FF5733")
st.markdown("Missing hash: `FF5733`")
st.markdown("Only 5 digits: `#FF573`")
st.markdown("7 digits: `#FF57333`")
st.markdown("Invalid hex: `#GG5733`")
st.markdown("Lowercase invalid: `#gg5733`")

st.header("With Other Markdown Elements")
st.markdown("**Bold text** with color `#FF0000` and *italic*")
st.markdown("`code inline` vs color `#00FF00` comparison")

st.header("In Tables")
st.markdown("""
| Color Name | Hex Code | Preview |
|------------|----------|---------|
| Red        | `#FF0000` | Red |
| Green      | `#00FF00` | Green |
| Blue       | `#0000FF` | Blue |
""")

st.header("With Links")
st.markdown("Check out [this color `#FF5733`](https://example.com)")

st.header("In Blockquotes")
st.markdown("""
> Our primary color is `#FF5733`
> and our secondary is `#00FF00`
""")

st.header("Multiple on Same Line")
st.markdown("Colors: `#FF0000`, `#00FF00`, `#0000FF`, `#FFFF00`, `#FF00FF`, `#00FFFF`")

st.header("Common Web Colors")
colors = {
    "Black": "#000000",
    "White": "#FFFFFF",
    "Red": "#FF0000",
    "Lime": "#00FF00",
    "Blue": "#0000FF",
    "Yellow": "#FFFF00",
    "Cyan": "#00FFFF",
    "Magenta": "#FF00FF",
    "Silver": "#C0C0C0",
    "Gray": "#808080",
    "Maroon": "#800000",
    "Olive": "#808000",
    "Green": "#008000",
    "Purple": "#800080",
    "Teal": "#008080",
    "Navy": "#000080",
}

for name, hex_code in colors.items():
    st.markdown(f"**{name}**: `{hex_code}`")

st.header("Color Palette Examples")

st.subheader("Material Design")
st.markdown("""
- Red 500: `#F44336`
- Pink 500: `#E91E63`
- Purple 500: `#9C27B0`
- Deep Purple 500: `#673AB7`
- Indigo 500: `#3F51B5`
- Blue 500: `#2196F3`
- Light Blue 500: `#03A5E8`
- Cyan 500: `#00BCD4`
- Teal 500: `#009688`
- Green 500: `#4CAF50`
- Light Green 500: `#8BC34A`
- Lime 500: `#CDDC39`
- Yellow 500: `#FFEB3B`
- Amber 500: `#FFC107`
- Orange 500: `#FF9800`
- Deep Orange 500: `#FF5722`
""")

st.subheader("Tailwind CSS")
st.markdown("""
- Slate 500: `#64748B`
- Gray 500: `#6B7280`
- Zinc 500: `#71717A`
- Neutral 500: `#737373`
- Stone 500: `#78716C`
- Red 500: `#EF4444`
- Orange 500: `#F97316`
- Amber 500: `#F59E0B`
- Yellow 500: `#EAB308`
- Lime 500: `#84CC16`
- Green 500: `#22C55E`
- Emerald 500: `#10B981`
- Teal 500: `#14B8A6`
- Cyan 500: `#06B6D4`
- Sky 500: `#0EA5E9`
- Blue 500: `#3B82F6`
- Indigo 500: `#6366F1`
- Violet 500: `#8B5CF6`
- Purple 500: `#A855F7`
- Fuchsia 500: `#D946EF`
- Pink 500: `#EC4899`
- Rose 500: `#F43F5E`
""")

st.header("With st.write()")
st.write("Testing with st.write: `#FF5733`")

st.header("With st.caption()")
st.caption("Caption with color: `#00FF00`")

st.header("In Columns")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("Red: `#FF0000`")
with col2:
    st.markdown("Green: `#00FF00`")
with col3:
    st.markdown("Blue: `#0000FF`")

st.header("In Expander")
with st.expander("Color Palette"):
    st.markdown("`#FF5733` `#33FF57` `#3357FF`")

st.header("In Tabs")
tab1, tab2 = st.tabs(["Colors 1", "Colors 2"])
with tab1:
    st.markdown("Primary: `#FF5733`")
with tab2:
    st.markdown("Secondary: `#33FF57`")

st.header("With Other Colored Text")
st.markdown("Regular text with :red[red colored text] and `#FF0000` hex swatch")
st.markdown(":blue[Blue text] with `#0000FF` swatch and :green-background[green background]")

st.header("Gradients (Edge Case)")
st.markdown("Gradient start `#FF0000` to end `#0000FF`")

st.success("✅ All tests rendered! Check if hex swatches appear correctly.")
