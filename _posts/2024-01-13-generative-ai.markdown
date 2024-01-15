---
layout: post
title: Generative AI Interfaces
description: Seven recurring design elements found in Generative AI applications and experiments.
image: /assets/images/generative-ai-interfaces/splash.png
authors: [tlyleung]
permalink: generative-ai-interfaces
---

# Introduction

In the rapidly evolving digital landscape, Generative AI emerges as a significant disruptor. Moving beyond the traditional relationship between user interaction cause and effect, it introduces an element of randomness and creativity to the output. Moving these tools from research experiments to integral components of our digital toolkit requires a fundamental rethinking of how humans and machines interact, posing critical questions about the future of interface design.

# Design Elements

Generative AI's unpredictable nature introduces new design challenges, requiring a fine balance between control, creativity, and usability. These challenges are predominantly shaped by the current limitations of Generative AI:

- **Slow Speed:** The autoregressive nature of language models and the iterative refinement required by diffusion models leads to slow outputs.

- **Limited Modalities:** Most interfaces are constrained by modality-specific tools like textboxes for text, paintbrush masks for images, and waveform selectors for audio.

- **Alignment issues:** While generating variations might not indicate a failure of alignment as it can used for preference refinement, it does highlight the need for improvement, particularly in areas like arithmetic, text display, and ensuring consistency across generations.

In this section we identify seven recurring design elements found in Generative AI applications and experiments.

## Audio Editors

Users are presented with a straightforward interface: a textbox for input and an audio waveform selection tool for masking. This setup supports a range of tasks, including text-to-audio conversions, where users can generate styled speech or sound effects, and audio-to-audio transformations guided by text prompts. This could involve restyling existing audio or infilling missing segments.

### Examples

<div>
    <div class="row">
        <div class="col-md-6">
            <figure class="figure bg-transparent">
                <img src="/assets/images/generative-ai-interfaces/lyria.png" class="figure-img img-fluid" alt="Google DeepMind Lyria screenshot">
                <figcaption class="figure-caption">Google DeepMind Lyria</figcaption>
            </figure>
        </div>
        <div class="col-md-6">
            <figure class="figure bg-transparent">
                <img src="/assets/images/generative-ai-interfaces/audiobox.png" class="figure-img img-fluid" alt="Meta Audiobox screenshot">
                <figcaption class="figure-caption">Meta Audiobox</figcaption>
            </figure>
        </div>
    </div>
</div>


## Conversational Interfaces

Conversational interfaces bring a dynamic and contextual aspect to user interaction. These interfaces can operate independently or in conjunction with other modalities, providing tailored responses and modifications based on the ongoing conversation. The chat interface is a natural extension of large language models and is a common, if overused, design element.

### Examples

<div>
    <div class="row">
        <div class="col-md-6">
            <figure class="figure bg-transparent">
                <img src="/assets/images/generative-ai-interfaces/copilot.png" class="figure-img img-fluid" alt="GitHub Copilot screenshot">
                <figcaption class="figure-caption">GitHub Copilot</figcaption>
            </figure>
        </div>
        <div class="col-md-6">
            <figure class="figure bg-transparent">
                <img src="/assets/images/generative-ai-interfaces/chatgpt.png" class="figure-img img-fluid" alt="OpenAI ChatGPT screenshot">
                <figcaption class="figure-caption">OpenAI ChatGPT</figcaption>
            </figure>
        </div>
    </div>
</div>


## Dashboards

Dashboards provide extensive control over output generation. For language models, these may include settings like temperature, top P, frequency penalty, presence penalty, and stop sequences. Image models might offer controls over dimensions, iterations, and prompt token weights. While powerful, dashboards risk overwhelming users.

### Examples

<div>
    <div class="row">
        <div class="col-md-6">
            <figure class="figure bg-transparent">
                <img src="/assets/images/generative-ai-interfaces/automatic1111.png" class="figure-img img-fluid" alt="Automatic1111 screenshot">
                <figcaption class="figure-caption">Automatic1111</figcaption>
            </figure>
        </div>
        <div class="col-md-6">
            <figure class="figure bg-transparent">
                <img src="/assets/images/generative-ai-interfaces/invoke.png" class="figure-img img-fluid" alt="Invoke screenshot">
                <figcaption class="figure-caption">Invoke</figcaption>
            </figure>
        </div>
    </div>
</div>

## Discord Servers

The use of Discord servers is predominantly used for image and video models, where a Discord Bot handles a variety of generative tasks such as media generation, creating variations, remixing media, and spatial manipulations like panning, zooming, and rotation. While this approach aligns well with the asynchronous nature of generation tasks that require longer processing times, its text-based command structure can be a barrier for less technical users. 

### Examples

<div>
    <div class="row">
        <div class="col-md-6">
            <figure class="figure bg-transparent">
                <img src="/assets/images/generative-ai-interfaces/midjourney.png" class="figure-img img-fluid" alt="Midjourney screenshot">
                <figcaption class="figure-caption">Midjourney</figcaption>
            </figure>
        </div>
        <div class="col-md-6">
            <figure class="figure bg-transparent">
                <img src="/assets/images/generative-ai-interfaces/pika.png" class="figure-img img-fluid" alt="Pika screenshot">
                <figcaption class="figure-caption">Pika</figcaption>
            </figure>
        </div>
    </div>
</div>

## Image Editors

Similar to audio editors but focused on visual content, image editors in Generative AI applications offer tools for creation, editing, and extension of images through text prompts and masking. In certain applications, users can even create an image from scratch by sketching with a paintbrush, similar to Google's Quick, Draw!

### Examples

<div>
    <div class="row">
<!--         <div class="col-md-6">
            <figure class="figure bg-transparent">
                <img src="/assets/images/generative-ai-interfaces/dalle2.png" class="figure-img img-fluid" alt="Open AI DALL·E 2 screenshot">
                <figcaption class="figure-caption">OpenAI DALL·E 2</figcaption>
            </figure>
        </div> -->
        <div class="col-md-6">
            <figure class="figure bg-transparent">
                <img src="/assets/images/generative-ai-interfaces/firefly.png" class="figure-img img-fluid" alt="Adobe Firefly screenshot">
                <figcaption class="figure-caption">Adobe Firefly</figcaption>
            </figure>
        </div>
        <div class="col-md-6">
            <figure class="figure bg-transparent">
                <img src="/assets/images/generative-ai-interfaces/runway.png" class="figure-img img-fluid" alt="Runway screenshot">
                <figcaption class="figure-caption">Runway</figcaption>
            </figure>
        </div>
    </div>
</div>

## Node-based Interfaces

Borrowing the concept from graphics packages like Blender and Houdini, node-based interfaces in Generative AI offer a visual approach to constructing workflows. Users can define and adjust parameters for each step of the process, linking different nodes to create an end-to-end workflow. This design element strikes a balance for users seeking the power of scripting without losing the intuitiveness of a visual interface.

### Examples

<div>
    <div class="row">
        <div class="col-md-6">
            <figure class="figure bg-transparent">
                <img src="/assets/images/generative-ai-interfaces/comfyui.png" class="figure-img img-fluid" alt="Comfy UI screenshot">
                <figcaption class="figure-caption">Comfy UI</figcaption>
            </figure>
        </div>
        <div class="col-md-6">
            <figure class="figure bg-transparent">
                <img src="/assets/images/generative-ai-interfaces/omnitool.png" class="figure-img img-fluid" alt="Omnitool screenshot">
                <figcaption class="figure-caption">Omnitool</figcaption>
            </figure>
        </div>
    </div>
</div>

## Templates

Templates serve as pre-defined prompts for various tasks. These templates can address a broad range of functions – from grammar correction, code explanation, and summarization in text-based tasks, to image outfilling, frame interpolation, and color grading in visual tasks, or even audio cleaning and transcription in audio tasks. As one of the simpler design elements, templates can reduce the user's cognitive load and task paralysis by presenting the user with a number of immediate use cases.

### Examples

<div>
    <div class="row">
        <div class="col-md-6">
            <figure class="figure bg-transparent">
                <img src="/assets/images/generative-ai-interfaces/jasper.png" class="figure-img img-fluid" alt="Jasper screenshot">
                <figcaption class="figure-caption">Jasper</figcaption>
            </figure>
        </div>
        <div class="col-md-6">
            <figure class="figure bg-transparent">
                <img src="/assets/images/generative-ai-interfaces/notion.png" class="figure-img img-fluid" alt="Notion screenshot">
                <figcaption class="figure-caption">Notion</figcaption>
            </figure>
        </div>
    </div>
</div>


# Looking to the Future

What would a user interface look like if these limitations of Generative AI models are solved:

- **Real-time Speed:** AI researchers are working to decrease inference speed through a combination of hardware optimization (like quantization and memory offloading), and model architecture improvements (like training student models and low-rank constraints). Models that can achieve output generation in times comparable to human reaction (around 250ms) could radically transform user experience, making interactions with AI almost indistinguishable from real-time human interactions.

- **Multimodality:** Byte-level models capable of processing and outputting arbitrary byte sequences would be modality agnostic and could revolutionize the way we interact with AI, enabling a seamless integration of different data types.

- **Perfect Alignment:** Work is being done on incorporating human feedback into model training. AI models achieving perfect alignment with human values and intentions could blur the interface between humans and computers.

There would be significant changes across many different areas:

- **Enhanced Creative Production Tools:** Future models will not only render final outputs but also provide editable source files. For example, in film production, the model could generate a complete movie scene complete with all the necessary components like rigging, lighting, and animation. Users would then be able to modify these elements using existing editors that can manipulate the source files.

- **Blurring the Line Between Media Producers and Consumers:** The distinction between creators and audiences in media consumption could become increasingly blurred. Imagine interactive experiences where the narrative, environment, and characters dynamically change in response to user interactions in real time. These narratives can be as realistic or fantastical as desired, with the inevitable seamless integration of product placements.

- **Communication and Decision Making:** AI models could stand in for individuals in both sending and receiving communications. This means tasks like scheduling, collaborating on projects, and negotiating contracts could be efficiently managed by AI, perfectly aligned with the user’s intentions and preferences. However, this level of delegation might lead to unexpected challenges, such as a sense of cognitive dissonance during in-person interactions, particularly if there's a mismatch between a person's actions and their intentions.
