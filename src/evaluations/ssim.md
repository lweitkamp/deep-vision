# Structural SIMilarity (SSIM) Index
SSIM is an attempt at measuring local structural differences of images instead of pixel-wise differences seen in [MSE](mse.md) and [PSNR](psnr.md).
The main philosophy is described as[^2]:

> The main function of the human eyes is to extract structural information from the viewing field.
> and the human visual system is highly adapted for this purpose.
> Therefore, a measurement of structural distortion should be a good approximation of perceived image distortion.

A function \\( S \\) is constructed to measure the *structural* similarity between two images, and this function should be somewhat invariant to pixel-level variation.
Here, \\(S\\) has the following qualities given two images \\(x\\) and \\(y\\):

1. \\(S\\) is symmetrical, \\(S(x, y) = S(y, x)\\).
2. \\(S\\) is Bounded, \\(0 \leq S(x, y) \leq 1\\).
3. \\(S\\) has a unique maximum, \\(S(x, y) = 1 \ \Longleftrightarrow \ x = y\\).

Note the following when using SSIM:

- SSIM provides complementary information to MSE and PSNR, it's not a replacement.
- SSIM is not going to help when the image is warped etc, since the assumption is that *local* structures are similar.
- SSIM uses a kernel of size 11, so keep this in mind as a receptive field of the measurement.

# SSIM In Depth
To get to the underlying structure, the similarity function \\(S\\) is a composition of three functions: luminance \\(l\\), contrast \\(c\\), and structure \\(s\\)[^1]:

\\[
\begin{eqnarray} 
S(x, y)
&=& f(l(x, y), c(x, y), s(x, y)) \\\\\\
&=& [l(x, y)]^\alpha \cdot [c(x, y)]^\beta \cdot [s(x, y)]^\gamma \\\\\\
&=&   \large [ \small \frac{2 \mu_x \mu_y + C_1}{\mu_x^2 + \mu_y^2 + C_1} \large ]^\alpha \small
\cdot \large [ \small \frac{2 \sigma_x \sigma_y + C_2}{ \sigma_x^2 + \sigma_y^2 + C_2} \large ]^\beta \small
\cdot \large [ \small \frac{\sigma_{xy} + C_3}{\sigma_x \sigma_y + C_3} \large ]^\gamma \small \\\\\\
\end{eqnarray}
\\]


Where we set \\(\alpha = \beta = \gamma = 1 \\) and \\(C_3 = \frac{C_2}{2} \\). The constants \\(C_1\\) and \\(C_2\\) are added for numerical stability and depend on the dynamic range of the input image. In the default implementation they are set as \\(C_1 = (0.01 * \text{dynamic_range})^2\\) and \\(C_2 = (0.03 * \text{dynamic_range})^2\\).

To calculate \\( \mu_{x}, \ \mu_{y}, \ \mu_{xy}  \\) and \\( \sigma_{x}, \ \sigma_{y}, \ \sigma_{xy}  \\), we will have to look at how \\( S \\) is built up from the three functions below.

## Luminance
Luminance is defined as the mean intensity of the image.
If we want to calculate the local 2D luminance, a Gaussian smoothing/blurring convolution is a good candidate for extracting it.


```python
:::from functools import partial
:::
:::import jax
:::import jax.numpy as jnp
:::
def gaussian_kernel(kernel_size: int, sigma: float) -> jax.Array:
:::    """Create a 2D Gaussian kernel."""
:::    assert kernel_size % 2 == 1, "Kernel size must be odd."
    x = jnp.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    y = x[:, jnp.newaxis]
    kernel = jnp.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / jnp.sum(kernel)

:::@partial(jax.jit, static_argnums=(1, 2))
def apply_gaussian_blur(
    image: jax.Array, dims: int, kernel_size: int, sigma: float
) -> jax.Array:
:::    """Apply Gaussian blur to a batch of images with multiple channels."""
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel_4d = kernel.reshape(kernel_size, kernel_size, 1, 1)

:::    if dims != 1:
:::        kernel_4d = jnp.repeat(kernel_4d, dims, axis=3)
:::
    out = jax.lax.conv_general_dilated(
        lhs=image,
        rhs=kernel_4d,
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=dims,
        precision=jax.lax.Precision.HIGHEST,
    )
    return out
```
This makes calculating the luminance quite easy:

\\[
\begin{eqnarray} 
\mu_x &=& \text{apply_gaussian_blur(x)} \\\\\\
\mu_y &=& \text{apply_gaussian_blur(y)} \\\\\\
l(x, y) &=& \frac{2 \mu_x \mu_y + C1}{\mu_x^2 + \mu_y^2 + C1}
\end{eqnarray}
\\]

Visually, we can 


## Contrast




## Structure



## In Practice
We can simplify the equation for \\( S \\) quite a bit and calculate it as follows:

\\[
\begin{eqnarray} 
S(x, y)
&=&   \large [ \small \frac{2 \mu_x \mu_y + C_1}{\mu_x^2 + \mu_y^2 + C_1} \large ]^\alpha \small
\cdot \large [ \small \frac{2 \sigma_x \sigma_y + C_2}{ \sigma_x^2 + \sigma_y^2 + C_2} \large ]^\beta \small
\cdot \large [ \small \frac{\sigma_{xy} + C_3}{\sigma_x \sigma_y + C_3} \large ]^\gamma \small \\\\\\
&=& \frac{(2 \mu_x \mu_y + C_1)(2 \sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
\end{eqnarray}
\\]


# References
[^2]: [Why is image quality assessment so difficult?](https://ieeexplore.ieee.org/document/5745362)

[^1]: The shape of these functions are hand-crafted to ensure the three properties remain.

