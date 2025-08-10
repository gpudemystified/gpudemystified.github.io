---
layout: default
title: Notes
permalink: /notes/
wide: true
---

<div class="notes-grid">
  {% assign notes = site.data.notes %}
  {% if notes and notes.size > 0 %}
    {% for note in notes %}
      <div class="notes-item">
        <div class="carousel" data-images='{{ note.images | jsonify | escape }}' data-title="{{ note.title | escape }}" data-text='{{ note.text | escape }}'>
          <div class="carousel-track" style="width: calc(100% * {{ note.images | size }})">
            {% for image in note.images %}
              <img class="carousel-slide" src="{{ image }}" alt="{{ note.title | escape }} - {{ forloop.index }}" />
            {% endfor %}
          </div>
          <button class="carousel-control prev" aria-label="Previous image">&#10094;</button>
          <button class="carousel-control next" aria-label="Next image">&#10095;</button>
        </div>
      </div>
    {% endfor %}
  {% else %}
    <p>No notes yet.</p>
  {% endif %}
</div>

<!-- Modal -->
<div class="notes-modal" id="notesModal" aria-hidden="true" role="dialog" aria-modal="true">
  <div class="notes-modal__content">
    <div class="notes-modal__media">
      <div class="carousel" id="modalCarousel">
        <div class="carousel-track" id="modalCarouselTrack"></div>
        <button class="carousel-control prev" id="modalPrev" aria-label="Previous image">&#10094;</button>
        <button class="carousel-control next" id="modalNext" aria-label="Next image">&#10095;</button>
      </div>
      <button class="notes-modal__close" id="modalClose" aria-label="Close">&#10005;</button>
    </div>
    <aside class="notes-modal__sidebar">
      <h2 id="modalTitle"></h2>
      <div id="modalText"></div>
    </aside>
  </div>
</div>

<script src="{{ '/assets/js/notes.js' | relative_url }}"></script>
