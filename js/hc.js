$(document).ready(function() {
    $(".container").hide();
    $(".container").fadeIn('5000');
    $(".showcase-wrapper").hide();
    $(".showcase-wrapper").fadeIn("slow");
});

$(function() {
    $('.project-box').click(function() {
        $(this).find('.post').slideToggle();
    });
});
