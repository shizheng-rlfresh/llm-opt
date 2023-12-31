<script>
    import { fade } from 'svelte/transition';
    import Nested from './Nested.svelte';
    import src from '$lib/images/favicon.png';
    let kkk = 'sveltekit';
</script>

<!--
    @component
    This is a page component.    
-->

<div class="flex min-h-screen flex-col items-center justify-center gap-4 p-4">
    <img class="w-16" {src} alt="svelte logo" />
    <h1 class="text-3xl font-bold underline"><a href="/">Welcome to {kkk}</a></h1>
    <p class="my-4">
        Visit <a href="https://kit.svelte.dev">kit.svelte.dev</a> to read the documentation
    </p>

    <input type="text" bind:value={kkk} />

    <Nested kit={kkk} />
    <Nested>
        {@const kkkk = `inputing ${kkk} ? sveltekit`}
        <p>{kkkk}</p>
    </Nested>

    {#key kkk}
        <div transition:fade>{kkk}</div>
    {/key}
</div>

<style lang="postcss">
    :global(html) {
        background-color: theme(colors.gray.100);
    }
</style>
