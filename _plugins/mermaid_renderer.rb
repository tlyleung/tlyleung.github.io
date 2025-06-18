require 'fileutils'
require 'digest'
require 'open3'
require 'securerandom'
require 'tmpdir'

module Jekyll
  class MermaidRenderer < Jekyll::Generator
    safe true

    CACHE_DIR = File.join(Dir.pwd, '.mermaid-cache')

    def generate(site)
      FileUtils.mkdir_p(CACHE_DIR)
      site.collections['notes'].docs.each { |note| process_page(note) }
      site.pages.each { |page| process_page(page) }
      site.posts.docs.each { |post| process_page(post) }
    end

    def process_page(page)
      content = page.content
      updated_content = content.gsub(/```mermaid(.*?)```/m) do |match|
        mermaid_code = $1.strip

        # Generate a hash of the Mermaid code to determine if it has changed
        hash = Digest::SHA256.hexdigest(mermaid_code)
        cached_light_path = File.join(CACHE_DIR, "#{hash}_light.svg")
        cached_dark_path = File.join(CACHE_DIR, "#{hash}_dark.svg")

        light_svg = cached_light_path if File.exist?(cached_light_path)
        dark_svg = cached_dark_path if File.exist?(cached_dark_path)

        unless light_svg && dark_svg
          Jekyll.logger.info "Regenerating Mermaid Diagram", "Hash: #{hash}"

          # Render light and dark versions
          light_svg_content = render_mermaid_to_svg(mermaid_code, 'neutral')
          dark_svg_content = render_mermaid_to_svg(mermaid_code, 'dark')

          if light_svg_content && dark_svg_content
            File.write(cached_light_path, light_svg_content)
            File.write(cached_dark_path, dark_svg_content)

            light_svg = cached_light_path
            dark_svg = cached_dark_path
          else
            Jekyll.logger.error "Mermaid Render Failed", "Could not render one or both themes for Mermaid diagram."
            next match # Skip rendering and return the original block
          end
        end

        # Combine the light and dark versions with proper class attributes
        light_svg_content = File.read(light_svg).gsub(/class="flowchart"/, 'class="flowchart block dark:hidden"')
        dark_svg_content = File.read(dark_svg).gsub(/class="flowchart"/, 'class="flowchart hidden dark:block"')

        <<~HTML
          #{light_svg_content}
          #{dark_svg_content}
        HTML
      end
    
      page.content = updated_content
    end

    def render_mermaid_to_svg(mermaid_code, theme)
      Dir.mktmpdir do |dir|
        input_path = File.join(dir, 'diagram.mmd')
        output_path = File.join(dir, 'diagram.svg')

        # Generate a unique ID for each diagram to replace default "my-svg"
        svg_id = "mermaid-#{SecureRandom.hex(4)}"

        File.write(input_path, mermaid_code)

        css_path = File.expand_path('assets/css/mermaid.css', Dir.pwd)
        command = "npx mmdc -i #{input_path} -o #{output_path} --svgId #{svg_id} --theme #{theme} --cssFile #{css_path} --backgroundColor transparent --quiet"
        stdout, stderr, status = Open3.capture3(command)

        if status.success?
          Jekyll.logger.info "Mermaid SVG Generated", output_path
          File.read(output_path)
        else
          Jekyll.logger.error "Mermaid Render Error", stderr
          nil
        end
      end
    end
  end
end
