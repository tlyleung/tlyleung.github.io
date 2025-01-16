require 'fileutils'
require 'open3'
require 'securerandom'
require 'tmpdir'

module Jekyll
  class MermaidRenderer < Jekyll::Generator
    safe true

    def generate(site)
      site.pages.each { |page| process_page(page) }
      site.posts.docs.each { |post| process_page(post) }
    end

    def process_page(page)
      content = page.content
      updated_content = content.gsub(/```mermaid(.*?)```/m) do |match|
        mermaid_code = $1.strip

        # Render light theme and replace the class
        light_svg = render_mermaid_to_svg(mermaid_code, 'neutral')
        light_svg.gsub!(/class="flowchart"/, 'class="flowchart block dark:hidden"')

        # Render dark theme and replace the class
        dark_svg = render_mermaid_to_svg(mermaid_code, 'dark')
        dark_svg.gsub!(/class="flowchart"/, 'class="flowchart hidden dark:block"')

        # Combine the light and dark versions
        <<~HTML
          #{light_svg}
          #{dark_svg}
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
